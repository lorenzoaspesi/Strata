# Gyro-M & Gyro-S Standardized Benchmark Procedure and Version Tracking

This document defines a reproducible, deterministic benchmark suite for **Qwen3.8-Flash-Next Gyro-M (`TQ2_0` / `TQ2_T`)** and **Gyro-S (`TQ1_0` / `TQK6`+`TQK7`)** on the local GTX 1070 (`sm_61`, 8 GiB VRAM) workstation, and tracks performance across engine and config revisions.

---

## 1. Hardware & Reference Environment

| Component | Specification |
| :--- | :--- |
| **GPU** | NVIDIA GeForce GTX 1070 8 GiB (`sm_61`, Pascal, CUDA 12.9) |
| **CPU** | Intel Core i7-7700 (4 cores / 8 threads, AVX2 + FMA) |
| **RAM / Bus** | 45.9 GiB DDR4, PCIe 3.0 x16 |
| **Storage** | `/mnt/ssd256` (NVMe/SSD for packs, GGUFs, DFlash2, disk session cache) |
| **Models** | Gyro-M (`Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf`, 29.88 GiB experts)<br>Gyro-S (`Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf`, 24.02 GiB experts) |
| **Drafter** | DFlash2 `qwen3.8-flash-next-dflash2.strata1.q4_0-selq4.pack` (`--spec 6 --draft-conf 0.7`, `372 MiB` VRAM) |
| **KV & Vision** | `--max-context 262144 --kv int8 --kv-resident 8192 --vram-reserve-mib 384`, `--vision --vram-elastic` |
| **Env Flags** | `STRATA_HC_Q8=1`, `STRATA_NO_ARENA_THP=1`, `STRATA_PREFILL_WARM=0`, `STRATA_WINDOW_WARM=0` |

---

## 2. Standardized Prompts

All benchmark runs use the harness [tools/gyro_bench.py](../tools/gyro_bench.py) with deterministic greedy decoding (`temperature=0.0`, `top_p=1.0`, `top_k=1`, `seed=42`, `max_tokens=128`) and `--fresh` (prepends a per-run tag `[bench-run:<unix_ts>]` so prefix KV cache reuse is `0` and full prefill is always measured).

| ID | Prompt Tokens | Category | Exact Prompt / Construction |
| :--- | :---: | :--- | :--- |
| **P1** | `110` | Short Technical | *"Explain the difference between memory-mapped I/O and direct DMA transfers over PCIe 3.0 in Linux, and why pinned host memory is required for asynchronous GPU copies. Keep your explanation structured and concise."* |
| **P2** | `114` | Code Generation | *"Write a C++20 header-only ring buffer class template RingBuffer<T, N> with lock-free single-producer single-consumer push and pop using std::atomic acquire/release memory ordering. Include full implementation and brief comments."* |
| **P3** | `3,383` | Medium Prefill (~3.4K) | `build_long_spec(32)` in `tools/gyro_bench.py`: 32 deterministic numbered subsystem specification sections + *"Summarize the top 5 architectural bottlenecks in the specification above and propose one concrete mitigation for each."* |
| **P4** | `33,232` | Long Prefill (>32K + KV Streaming) | `build_long_spec(320)` in `tools/gyro_bench.py`: 320 deterministic numbered subsystem specification sections (>32K context, 4x larger than the 8,192-token VRAM KV ring so KV streaming from pinned RAM is exercised) + the same 5-bottleneck summary question. |

---

## 3. How to Replicate

### 3.1 Benchmark Live Gyro-M (`strata.service`)

```bash
# Ensure Gyro-M is running and ready
systemctl --user restart strata.service
/home/lori/.local/bin/strata runtime status --profile launch-gyro-m-256k

# Run full P1–P4 suite (reads api_key from launch JSON without printing it)
python3 tools/gyro_bench.py --model m --prompts P1,P2,P3,P4 --fresh
```

### 3.2 Benchmark Gyro-S (`launch-gyro-s-256k`)

```bash
# Stop Gyro-M so the 8 GiB GPU is free, start Gyro-S, benchmark, then restore Gyro-M
systemctl --user stop strata.service
STRATA_ENGINE_BACKEND=cuda /home/lori/.local/bin/strata runtime start --profile launch-gyro-s-256k
python3 tools/gyro_bench.py --model s --prompts P1,P2,P3,P4 --fresh
/home/lori/.local/bin/strata runtime stop --profile launch-gyro-s-256k
systemctl --user start strata.service
```

### 3.3 Metrics Recorded Per Run

The harness parses the engine log (`gyro-{m,s}-256k.engine.log`) after each request and reports:
- **Boot / VRAM:** `expert_slots`, `expert_cache_gib`, `prefill_chunk`, `vram_free_load_mib`
- **Per-Prompt:** `prompt_tokens` (`reused=0`), `prefill_tok_s`, `decode_tok_s` (`128` generated tokens), `cache_hit_pct`, `drafts_accepted/drafts_total`, and total `wall_s`.

---

## 4. Performance Tracking Across Versions

### 4.1 Gyro-M (`Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf`)

| Version / Commit | Config Notes | Expert Slots (VRAM) | Prefill Chunk | P1 (110t)<br>Prefill / Decode | P2 (114t)<br>Prefill / Decode | P3 (~3.4K)<br>Prefill / Decode | P4 (~33.2K)<br>Prefill / Decode |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.1.67.100` (pre-VRAM-opt) | `--kv-resident 20480`, BF16 draft head, BF16 DFlash2 KV | 701 (`0.85 GiB`) | 256 | — / `17.2 tok/s` *(legacy 74t)* | — | — | — |
| `0.1.69.100` (`2a40b2eb8`, baseline) | Tier 1 (`--kv-resident 8192`, warm=0) + indexed draft head + FP16 DFlash2 KV | 1,045 (`1.27 GiB`) | 1024 | `83.9` / `22.4 tok/s`<br>*(69.7% hit, 71/83)* | `82.5` / `25.6 tok/s`<br>*(66.7% hit, 91/103)* | `279.5` / `20.4 tok/s`<br>*(59.4% hit, 78/100)* | — |
| `0.1.69.100` + `rows_tqk_split` | Same Gyro-M config + `vision.exe` updated to `0.1.69.100` | 1,045 (`1.27 GiB`) | 1024 | `88.3` / `22.1 tok/s`<br>*(64.8% hit, 78/95)* | `82.4` / `26.5 tok/s`<br>*(66.2% hit, 92/98)* | `282.9` / `21.0 tok/s`<br>*(65.1% hit, 75/93)* | **`279.4` / `20.6 tok/s`**<br>*(59.9% hit, 82/103, 126.9s)* |

### 4.2 Gyro-S (`Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf`)

| Version / Commit | Config Notes | Expert Slots (VRAM) | Prefill Chunk | P1 (110t)<br>Prefill / Decode | P2 (114t)<br>Prefill / Decode | P3 (~3.4K)<br>Prefill / Decode | P4 (~33.2K)<br>Prefill / Decode |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `0.1.69.100` (`2a40b2eb8`, baseline) | `--kv-resident 20480`, suffix-only (`--spec 4`, no DFlash2), warm=1 | 2,494 (`2.44 GiB`) | 3328 | `104.9` / `22.9 tok/s`<br>*(80.6% hit, 27/30)* | `97.6` / `24.1 tok/s`<br>*(85.9% hit, 38/46)* | `317.8` / `20.4 tok/s`<br>*(84.8% hit, 7/14)* | — |
| `0.1.69.100` (Step A: config-only) | Added DFlash2 `selq4` (`--spec 6 --draft-conf 0.7`) + `--kv-resident 8192` + warm=0 | 2,309 (`2.26 GiB`) | 3072 | `105.4` / `25.1 tok/s`<br>*(79.7% hit, 72/84)* | `97.3` / `27.5 tok/s`<br>*(85.5% hit, 73/86)* | `304.1` / `25.7 tok/s`<br>*(81.3% hit, 82/101)* | — |
| `0.1.69.100` + `rows_tqk_split` | Step A config + bit-exact `TQK6`/`TQK7` AVX2 split-half + `NT_CONST` (`tq_avx2.cpp`) | 2,309 (`2.26 GiB`) | 3072 | **`109.2` / `25.4 tok/s`**<br>*(80.9% hit, 61/69)* | **`98.0` / `29.1 tok/s`**<br>*(84.4% hit, 82/93)* | **`308.8` / `28.4 tok/s`**<br>*(79.6% hit, 93/104)* | **`310.9` / `24.1 tok/s`**<br>*(80.4% hit, 80/106, 113.7s)* |

### 4.3 Summary of Net Gains (Zero Quality Impact)

- **Gyro-S (`0.1.69.100` baseline → current):**
  - **P1 decode:** `22.9 → 25.4 tok/s` (**+10.9%**), prefill `104.9 → 109.2 tok/s` (**+4.1%**)
  - **P2 decode:** `24.1 → 29.1 tok/s` (**+20.7%**)
  - **P3 (~3.4K) decode:** `20.4 → 28.4 tok/s` (**+39.2%**), total wall `16.85s → 15.50s`
  - **P4 (~33.2K):** `310.9 tok/s` prefill, `24.1 tok/s` decode with active KV streaming (`80.4%` expert hit)
- **Gyro-M (`0.1.69.100` baseline → current):**
  - Verified zero regression (`P1: 22.4 → 22.1`, `P2: 25.6 → 26.5`, `P3: 20.4 → 21.0 tok/s`).
  - **P4 (~33.2K):** `279.4 tok/s` prefill, `20.6 tok/s` decode with active KV streaming (`59.9%` expert hit).

---

## 5. CPU Kernel Micro-Benchmark & Ruled-Out Experiments

Pinned single-core (`taskset -c 0`, Intel Core i7-7700, `640 x 2560` matrix, bit-exact `memcmp == 0` vs scalar across `nt=1..8`):

| Kernel (`tq_avx2.cpp`) | `kb=6` (`TQK6`) `nt=1` / `nt=4` | `kb=7` (`TQK7`) `nt=1` / `nt=4` | Decision |
| :--- | :---: | :---: | :--- |
| Baseline `rows_t<KB>` | `0.262` / `0.467 ms` | `0.262` / `0.486 ms` | Baseline |
| Scalar state + `_mm_setr_epi32` + `vpgatherqq` | `0.303` / `0.510 ms` (-15% to -24%) | `0.312` / `0.506 ms` (-15% to -29%) | **Rejected** (GPR→XMM `vpinsrd` overhead) |
| `rows_tqk_split<KB, NT_CONST>` (`rows4` + split-half + `NT_CONST`) | **`0.259` / `0.450 ms`** (+1.2% / +3.8%) | **`0.264` / `0.451 ms`** (-0.8% / +7.8%) | **Kept** (bit-exact, faster at `nt=4`) |

Other evaluated zero-quality toggles:
- **`STRATA_Q8_INPLACE=1`:** **Rejected** on `sm_61` — triggers `non-finite logits (#879)` during `sm_61` prefill (in-place SoA repacking is incompatible with the prefill dense GEMM layout).
- **`STRATA_ADAPT_NOWAIT=1`:** **No-op** on `sm_61` when `STRATA_ADAPT_FETCH=2` is active (`the tier's own copies are off`).
- **Verify CUDA graph width cap below `1..7`:** **Rejected** — DFlash2 (`--spec 6`) dynamically uses widths `1..7`; evicting active widths causes per-step re-capture thrashing.
