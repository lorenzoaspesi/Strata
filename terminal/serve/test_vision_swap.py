from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
from pathlib import Path

from serve import server as srv
from serve.frontend import ChatTemplate


PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)


class VisionSwapTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="strata-vis-swap-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _write_fake_vision_exe(self, *, fail_first_wake_oom: bool = False) -> Path:
        script = self.root / "fake_strata_vision.py"
        log_file = self.root / "vision_events.log"
        script.write_text(
            "#!/usr/bin/env python3\n"
            + textwrap.dedent(
                f"""\
                import struct, sys
                from pathlib import Path

                log = Path({str(log_file)!r})
                def rec(msg: str) -> None:
                    with log.open("a", encoding="utf-8") as fh:
                        fh.write(msg + "\\n")

                swap = "--swap-vram" in sys.argv
                rec(f"START swap={{swap}}")
                if swap:
                    print("READY 16 own swap 1030", flush=True)
                else:
                    print("READY 16 own", flush=True)

                sleeping = swap
                failed_once = False
                for raw in sys.stdin:
                    line = raw.strip()
                    if not line:
                        continue
                    if line == "QUIT":
                        rec("QUIT")
                        break
                    if line == "WAKE":
                        if {fail_first_wake_oom!r} and not failed_once:
                            failed_once = True
                            rec("WAKE_OOM")
                            print("ERR GPU memory for vision wake: out of memory", flush=True)
                            continue
                        sleeping = False
                        rec("WAKE")
                        print("OK 71.5", flush=True)
                        continue
                    if line == "SLEEP":
                        sleeping = True
                        rec("SLEEP")
                        print("OK", flush=True)
                        continue
                    if line.startswith("ENC "):
                        _, rest = line.split(" ", 1)
                        img_name, out_name = rest.rsplit(" ", 1)
                        rec(f"ENC sleeping={{sleeping}}")
                        hdr = struct.pack("<5i", 0x31455653, 4, 2, 2, 16)
                        Path(out_name).write_bytes(hdr + (b"\\x00" * (4 * 16 * 4)))
                        print("OK 4 2 2 18", flush=True)
                        continue
                    print("ERR unknown command", flush=True)
                """
            ),
            encoding="utf-8",
        )
        script.chmod(0o755)
        return script

    def test_apply_vision_defaults_and_runner_preserve_swap_vram(self):
        cfg = {
            "exe": "component:strata-engine",
            "args": ["--max-context", "262144"],
            "vision": {
                "exe": "/tmp/strata-vision",
                "mmproj": "/tmp/mmproj.gguf",
                "model": "/tmp/model.gguf",
                "gpu": True,
                "cuda_device": 0,
                "swap_vram": True,
                "max_tokens": 1600,
            },
        }
        applied = srv.apply_vision_defaults(dict(cfg))
        self.assertIs(applied.get("vram_elastic"), True)
        self.assertIs(applied["vision"].get("swap_vram"), True)
        self.assertNotIn("elastic", applied["vision"])

        rargs = srv.Vision.reader_args(applied["vision"])
        self.assertIn("--gpu", rargs)
        self.assertIn("--swap-vram", rargs)

        from strata_cli.launch import runner, settings as S
        from strata_control import tune_knobs

        base = {
            "exe": "component:strata-engine",
            "args": ["--max-context", "262144"],
            "vision": {"exe": "/tmp/strata-vision", "mmproj": "/tmp/m.gguf", "model": "/tmp/t.gguf", "gpu": True},
        }
        prev = {
            "args": ["--max-context", "262144"],
            "vram_elastic": True,
            "vision": {"gpu": True, "swap_vram": True, "swap_mib": 1400},
        }
        ls = S.LaunchSettings(family="gyro", model="Gyro-S", access="lan", port=18931)
        derived = runner.derive_config(base, ls, bool_flags=tune_knobs.BOOL_FLAGS, prev=prev)
        self.assertIs(derived.get("vram_elastic"), True)
        self.assertIs(derived["vision"].get("swap_vram"), True)
        self.assertEqual(derived["vision"].get("swap_mib"), 1400)

    def test_swap_vram_shrinks_wakes_encodes_sleeps_and_restores_experts_before_prefill(self):
        fake_exe = self._write_fake_vision_exe()
        mmproj = self.root / "mmproj.gguf"
        model = self.root / "model.gguf"
        mmproj.write_bytes(b"GGUF")
        model.write_bytes(b"GGUF")

        events_log = self.root / "vision_events.log"
        tok = srv.ByteTokenizer()

        class ElasticTrackEngine(srv.MockEngine):
            def __init__(self):
                super().__init__(tok, ["caption ok"], max_context=65536)
                self.current_reserve = 0
                self.generate_reserves: list[int] = []

            def alive(self) -> bool:
                return True

            def vram(self, reserve_mib: int | None, timeout: float = 120.0) -> dict:
                r = 0 if reserve_mib is None else int(reserve_mib)
                self.current_reserve = r
                with events_log.open("a", encoding="utf-8") as fh:
                    fh.write(f"ENGINE_VRAM {r}\n")
                full = 2560
                cur = max(512, full - r)
                return {"expert_cache_mib": cur, "expert_cache_full_mib": full, "vram_free_mib": max(160, r)}

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                self.generate_reserves.append(self.current_reserve)
                with events_log.open("a", encoding="utf-8") as fh:
                    fh.write(f"GENERATE reserve={self.current_reserve}\n")
                yield from super().generate(ids, max_new, sampling, cancel, embeddings=embeddings)

        vision = srv.Vision(
            {
                "exe": str(fake_exe),
                "mmproj": str(mmproj),
                "model": str(model),
                "gpu": True,
                "swap_vram": True,
                "max_tokens": 1600,
            }
        )
        self.addCleanup(vision.unload)
        self.assertTrue(vision.swap_supported)
        self.assertTrue(vision.sleeping)
        self.assertEqual(vision.swap_mib, 1030 + 256)

        eng = ElasticTrackEngine()
        tpl = ChatTemplate(Path(srv.__file__).resolve().parent / "chat_template.jinja")
        svc = srv.Service(eng, tok, tpl, vision=vision)

        img_file = self.root / "pic1.png"
        img_file.write_bytes(PNG_1X1)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": str(img_file)},
                    {"type": "text", "text": "Describe this picture."},
                ],
            }
        ]

        ids, thinking, max_new = svc.prepare(messages, None, {}, max_new=32, cancel=threading.Event())
        # Right after prepare() — BEFORE Service.run() / generate() — vision is already asleep & experts restored!
        self.assertTrue(vision.sleeping)
        self.assertEqual(eng.current_reserve, 0)

        out_events = list(svc.run(ids, thinking, None, max_new, {}, threading.Event()))
        self.assertEqual(out_events[-1][0], "done")
        self.assertEqual(eng.generate_reserves, [0])

        recorded = events_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            recorded,
            [
                "START swap=True",
                "ENGINE_VRAM 1286",
                "WAKE",
                "ENC sleeping=False",
                "SLEEP",
                "ENGINE_VRAM 0",
                "GENERATE reserve=0",
            ],
        )

        # Turn 2 with the exact same image hits .sve disk cache -> ZERO VRAM shrink/wake/sleep!
        ids2, thinking2, max_new2 = svc.prepare(messages, None, {}, max_new=32, cancel=threading.Event())
        list(svc.run(ids2, thinking2, None, max_new2, {}, threading.Event()))
        recorded2 = events_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(recorded2[len(recorded):], ["GENERATE reserve=0"])

    def test_wake_oom_bumps_swap_mib_and_retries_cleanly(self):
        fake_exe = self._write_fake_vision_exe(fail_first_wake_oom=True)
        mmproj = self.root / "mmproj.gguf"
        model = self.root / "model.gguf"
        mmproj.write_bytes(b"GGUF")
        model.write_bytes(b"GGUF")

        tok = srv.ByteTokenizer()

        class VramRecorderEngine(srv.MockEngine):
            def __init__(self):
                super().__init__(tok, ["ok"], max_context=65536)
                self.vram_asks: list[int] = []

            def alive(self) -> bool:
                return True

            def vram(self, reserve_mib: int | None, timeout: float = 120.0) -> dict:
                r = 0 if reserve_mib is None else int(reserve_mib)
                self.vram_asks.append(r)
                return {"expert_cache_mib": 2560 - r, "expert_cache_full_mib": 2560}

        vision = srv.Vision(
            {
                "exe": str(fake_exe),
                "mmproj": str(mmproj),
                "model": str(model),
                "gpu": True,
                "swap_vram": True,
            }
        )
        self.addCleanup(vision.unload)
        eng = VramRecorderEngine()
        svc = srv.Service(eng, tok, ChatTemplate(Path(srv.__file__).resolve().parent / "chat_template.jinja"), vision=vision)

        img_file = self.root / "pic_oom.png"
        img_file.write_bytes(PNG_1X1)
        msgs = [{"role": "user", "content": [{"type": "image", "source": str(img_file)}, {"type": "text", "text": "Hi"}]}]
        svc.prepare(msgs, None, {}, max_new=16, cancel=threading.Event())
        self.assertEqual(eng.vram_asks, [1286, 1542, 0])
        self.assertTrue(vision.sleeping)


if __name__ == "__main__":
    unittest.main()
