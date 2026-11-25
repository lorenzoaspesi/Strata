#!/usr/bin/env python3
"""Standardized reproducible benchmark harness for Strata Gyro-M and Gyro-S.

Supports:
  --mode server : Benchmarks the live strata.service HTTP server (127.0.0.1:18931)
                  and parses exact engine metrics from the engine log.
  --mode cli    : Runs the standalone strata engine binary directly (requires
                  strata.service to be stopped so the GPU has full VRAM).

Never prints API keys.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import struct
import subprocess
import sys
import time
import urllib.request
import zlib
from pathlib import Path

LAUNCH_DIR = Path("/home/lori/.local/state/strata/launch")
RUNTIME_DIR = Path("/home/lori/.local/state/strata/runtime")

# Standardized deterministic prompts (P1: short tech, P2: code, P3: ~1.8K-tok prefill)
PROMPT_P1 = (
    "Explain the difference between memory-mapped I/O and direct DMA transfers over "
    "PCIe 3.0 in Linux, and why pinned host memory is required for asynchronous GPU "
    "copies. Keep your explanation structured and concise."
)

PROMPT_P2 = (
    "Write a C++20 header-only ring buffer class template RingBuffer<T, N> with "
    "lock-free single-producer single-consumer push and pop using std::atomic "
    "acquire/release memory ordering. Include full implementation and brief comments."
)


def build_long_spec(num_sections: int) -> str:
    """Build a deterministic technical specification prompt (~103 tokens/section)."""
    blocks = []
    subsystems = [
        ("Scheduler", "dispatches micro-batches across 48 transformer layers with dynamic priority queues"),
        ("ExpertCache", "manages a bounded LRU-pinned VRAM pool of quantized MoE experts with PCIe prefetch"),
        ("KVStreamer", "streams int8 key-value pages between pinned host RAM and an 8192-token VRAM ring"),
        ("DraftVerifier", "validates speculative token windows of width 1..8 via captured CUDA graphs"),
        ("RouterEngine", "computes top-10 expert scores across 512 experts per layer using BF16 GEMV"),
        ("HadamardStage", "applies block-128 Walsh-Hadamard rotations before trellis expert projection"),
        ("DMAEngine", "overlaps asynchronous H2D expert transfers with GPU dense attention kernels"),
        ("SessionStore", "persists prefix KV checkpoints to NVMe SSD with CRC32 integrity verification"),
    ]
    for i in range(1, num_sections + 1):
        name, desc = subsystems[(i - 1) % len(subsystems)]
        blocks.append(
            f"Section {i:03d} ({name}-v{(i % 4) + 1}): This module {desc}. "
            f"Under peak load, stage {i} observes {120 + (i * 17) % 90} us median latency, "
            f"a p99 tail of {340 + (i * 29) % 210} us, and transfers {16 + (i * 7) % 48} MiB/s "
            f"across the PCIe 3.0 x16 interconnect while maintaining {(i * 13) % 30 + 62}% cache "
            f"locality across 4 worker threads and 15 streaming multiprocessors."
        )
    body = "\n".join(blocks)
    return (
        "Analyze the following system architecture specification:\n\n"
        f"{body}\n\n"
        "Summarize the top 5 architectural bottlenecks in the specification above "
        "and propose one concrete mitigation for each."
    )


PROMPTS = {
    "P1": PROMPT_P1,
    "P2": PROMPT_P2,
    "P3": build_long_spec(32),    # ~3.3K tokens
    "P4": build_long_spec(320),   # ~33K tokens (>32K context, exercises KV streaming)
    "V1": "Describe the geometric shapes and colors in this image concisely.",
}


def make_test_png(width: int = 1280, height: int = 720, nonce: str = "") -> bytes:
    """Generate a deterministic 1280x720 RGB PNG with colored geometric blocks.

    When `nonce` is non-empty, encodes it in the top-left 4 pixels so the server's
    .sve disk image cache is bypassed and the GPU vision encoder always runs.
    """
    seed = int(nonce) & 0xFFFFFFFF if nonce.isdigit() else 0
    rows = []
    for y in range(height):
        row = bytearray(1 + width * 3)  # filter byte 0 + RGB
        if y < height // 2:
            # Left half red, right half blue
            for x in range(width):
                off = 1 + x * 3
                if x < width // 2:
                    row[off], row[off + 1], row[off + 2] = 210, 45, 45
                else:
                    row[off], row[off + 1], row[off + 2] = 45, 90, 210
        else:
            # Left half green, right half yellow
            for x in range(width):
                off = 1 + x * 3
                if x < width // 2:
                    row[off], row[off + 1], row[off + 2] = 45, 190, 75
                else:
                    row[off], row[off + 1], row[off + 2] = 235, 205, 40
        if y == 0 and seed:
            row[1] = (seed >> 0) & 0xFF
            row[2] = (seed >> 8) & 0xFF
            row[3] = (seed >> 16) & 0xFF
            row[4] = (seed >> 24) & 0xFF
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, level=1)) + chunk(b"IEND", b"")


RE_SERVE_LINE = re.compile(
    r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
    r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)"
)
RE_HIT_LINE = re.compile(
    r"strata serve: decode expert cache hit rate: ([\d.]+)% \((\d+) hits / (\d+) lookups\)"
)
RE_CACHE_SLOTS = re.compile(r"expert cache (\d+) slots, ([\d.]+) GiB of VRAM")
RE_CHUNK = re.compile(r"prompt chunk auto: (\d+) tokens")
RE_VRAM_FREE = re.compile(r"strata serve: (\d+) MiB of VRAM free with everything loaded")
RE_VISION_SWAP = re.compile(
    r"vision weights swapped in \((\d+) ms\) and out of VRAM; expert cache restored \((\d+) of (\d+) MiB\)"
)
RE_VRAM_DRAIN = re.compile(
    r"strata serve: VRAM (\d+) MiB kept free on each card: the expert caches hold (\d+) of (\d+) slots .*?CUDA0 (\d+) MiB free now"
)


def load_launch_config(model: str) -> dict:
    cfg_path = LAUNCH_DIR / f"gyro-{model}-256k.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_engine_log_tail(log_path: Path, start_offset: int) -> dict:
    if not log_path.exists():
        return {}
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(start_offset)
        text = f.read()
    info: dict = {}
    for m in RE_SERVE_LINE.finditer(text):
        info["prompt_tokens"] = int(m.group(1))
        info["reused_tokens"] = int(m.group(2))
        info["read_tokens"] = int(m.group(3))
        info["prefill_ms"] = int(m.group(4))
        info["prefill_tok_s"] = float(m.group(5))
        info["gen_tokens"] = int(m.group(6))
        info["decode_ms"] = int(m.group(7))
        info["decode_tok_s"] = float(m.group(8))
        info["drafts_accepted"] = int(m.group(9))
        info["drafts_total"] = int(m.group(10))
    for m in RE_HIT_LINE.finditer(text):
        info["cache_hit_pct"] = float(m.group(1))
    for m in RE_VRAM_DRAIN.finditer(text):
        if int(m.group(1)) > 0:
            info["vision_drain_slots"] = int(m.group(2))
            info["vision_drain_free_mib"] = int(m.group(4))
    return info


def parse_server_log_vision(server_log: Path, start_offset: int) -> dict:
    if not server_log.exists():
        return {}
    with server_log.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(start_offset)
        text = f.read()
    info: dict = {}
    for m in RE_VISION_SWAP.finditer(text):
        info["vision_swap_ms"] = int(m.group(1))
        info["vision_restored_mib"] = int(m.group(2))
        info["vision_full_mib"] = int(m.group(3))
    return info


def parse_boot_metrics(log_path: Path) -> dict:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    # Take the last boot section
    info: dict = {}
    for m in RE_CACHE_SLOTS.finditer(text):
        info["expert_slots"] = int(m.group(1))
        info["expert_cache_gib"] = float(m.group(2))
    for m in RE_CHUNK.finditer(text):
        info["prefill_chunk"] = int(m.group(1))
    for m in RE_VRAM_FREE.finditer(text):
        info["vram_free_load_mib"] = int(m.group(1))
    return info


def run_server_prompt(
    port: int,
    api_key: str,
    model_name: str,
    prompt_id: str,
    prompt_text: str,
    max_tokens: int,
    log_path: Path,
    server_log: Path,
    nonce: str,
) -> dict:
    # Add a unique prefix comment only if requested to bust prefix KV cache
    text = f"[bench-run:{nonce}]\n{prompt_text}" if nonce else prompt_text
    if prompt_id == "V1":
        png_b64 = base64.b64encode(make_test_png(1280, 720, nonce=nonce)).decode("ascii")
        user_content: object = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
            {"type": "text", "text": text},
        ]
        eff_max_tokens = min(max_tokens, 64)
    else:
        user_content = text
        eff_max_tokens = max_tokens
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "seed": 42,
        "max_tokens": eff_max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    start_off = log_path.stat().st_size if log_path.exists() else 0
    srv_off = server_log.stat().st_size if server_log.exists() else 0
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    wall_s = time.perf_counter() - t0
    time.sleep(0.3)
    log_info = parse_engine_log_tail(log_path, start_off)
    vis_info = parse_server_log_vision(server_log, srv_off) if prompt_id == "V1" else {}
    usage = body.get("usage", {})
    return {
        "prompt_id": prompt_id,
        "wall_s": round(wall_s, 2),
        "usage_prompt_tokens": usage.get("prompt_tokens"),
        "usage_completion_tokens": usage.get("completion_tokens"),
        **log_info,
        **vis_info,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["m", "s"], required=True)
    ap.add_argument("--prompts", default="P1,P2,P3,P4,V1")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--fresh", action="store_true", help="Add unique run tag to prevent prefix cache reuse")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    cfg = load_launch_config(args.model)
    log_path = Path(cfg["log"])
    server_log = RUNTIME_DIR / f"launch-gyro-{args.model}-256k.server.log"
    boot = parse_boot_metrics(log_path)
    nonce = str(int(time.time())) if args.fresh else ""

    results = []
    for pid in [p.strip() for p in args.prompts.split(",") if p.strip()]:
        res = run_server_prompt(
            port=cfg["port"],
            api_key=cfg["api_key"],
            model_name=cfg["model_name"],
            prompt_id=pid,
            prompt_text=PROMPTS[pid],
            max_tokens=args.max_tokens,
            log_path=log_path,
            server_log=server_log,
            nonce=nonce,
        )
        results.append(res)
        extra = ""
        if pid == "V1":
            extra = (
                f" | vision_swap={res.get('vision_swap_ms','?')}ms"
                f" drain_slots={res.get('vision_drain_slots','?')}"
                f" drain_free={res.get('vision_drain_free_mib','?')}MiB"
                f" restored={res.get('vision_restored_mib','?')}/{res.get('vision_full_mib','?')}MiB"
            )
        print(
            f"{pid}: prompt={res.get('prompt_tokens','?')}t "
            f"(reused={res.get('reused_tokens','?')}, read={res.get('read_tokens','?')}) "
            f"prefill={res.get('prefill_tok_s','?')} tok/s | "
            f"decode={res.get('decode_tok_s','?')} tok/s ({res.get('gen_tokens','?')}t) | "
            f"hit={res.get('cache_hit_pct','?')}% | "
            f"drafts={res.get('drafts_accepted','?')}/{res.get('drafts_total','?')} | "
            f"wall={res['wall_s']}s{extra}",
            flush=True,
        )

    summary = {"model": f"gyro-{args.model}", "boot": boot, "results": results}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
