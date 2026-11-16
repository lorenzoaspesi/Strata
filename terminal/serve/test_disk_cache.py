from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from serve.disk_cache import DiskSessionCache, lcp_len


class FakeSessionRefused(Exception):
    def __init__(self, message: str, *, kind: str = "invalid", published: bool = False, status: int = 500):
        super().__init__(message)
        self.kind = kind
        self.published = published
        self.status = status


class FakeEngine:
    def __init__(self, *, bytes_per_token: int = 64):
        self.gen = 1
        self.batch = 0
        self.session_save_reclaim = False
        self.bytes_per_token = bytes_per_token
        self.held_tokens: list[int] = []
        self.calls: list[tuple[str, str]] = []
        self.fail_restore_paths: dict[str, Exception] = {}
        self.fail_next_save: Exception | None = None
        self._alive = True

    def alive(self) -> bool:
        return self._alive

    def session_file(self, action: str, path: str) -> dict:
        self.calls.append((action, path))
        if action == "save":
            if self.fail_next_save is not None:
                err, self.fail_next_save = self.fail_next_save, None
                raise err
            payload = json.dumps({"tokens": self.held_tokens}).encode("utf-8")
            target_len = max(len(payload), len(self.held_tokens) * self.bytes_per_token)
            blob = payload + b"\n" + (b"x" * max(0, target_len - len(payload) - 1))
            Path(path).write_bytes(blob)
            return {"tokens": len(self.held_tokens), "bytes": len(blob), "ms": 4.5}
        if action == "restore":
            if path in self.fail_restore_paths:
                raise self.fail_restore_paths[path]
            raw = Path(path).read_bytes()
            header = raw.split(b"\n", 1)[0]
            data = json.loads(header.decode("utf-8"))
            self.held_tokens = list(data["tokens"])
            return {"tokens": len(self.held_tokens), "bytes": len(raw), "ms": 3.2}
        raise ValueError(f"unknown action {action!r}")


class FakeService:
    def __init__(self, engine: FakeEngine | None = None):
        self.engine = engine or FakeEngine()
        self.fifo = threading.Lock()
        self.status_lock = threading.Lock()
        self.status = {"busy": False, "queued": 0}
        self._loaded = True

    def loaded(self) -> bool:
        return self._loaded and self.engine.alive()

    def run_turn(self, cache: DiskSessionCache, prompt: list[int], generated: list[int], *, ok: bool = True):
        with self.fifo:
            restored = cache.before_generate(self, prompt)
            if ok:
                self.engine.held_tokens = list(prompt) + list(generated)
            cache.after_generate(self, list(prompt) + list(generated), ok=ok)
            return restored


class DiskCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="strata-dc-test-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_blank_or_zero_budget_disables_cleanly(self):
        self.assertEqual(lcp_len([1, 2, 3, 4], [1, 2, 9]), 2)
        c1 = DiskSessionCache.from_config("", 64)
        self.assertFalse(c1.enabled)
        self.assertIsNone(c1.dir)

        c2 = DiskSessionCache.from_config(None, 64)
        self.assertFalse(c2.enabled)

        c3 = DiskSessionCache.from_config(str(self.root / "cache"), 0)
        self.assertFalse(c3.enabled)
        self.assertFalse((self.root / "cache").exists())

        with self.assertRaises(ValueError):
            DiskSessionCache.from_config(str(self.root / "cache"), -1)
        with self.assertRaises(ValueError):
            DiskSessionCache.from_config(str(self.root / "cache"), True)
        with self.assertRaises(ValueError):
            DiskSessionCache.from_config("//server/share/cache", 64)

    def test_unwritable_or_missing_parent_disables_without_creating_parents(self):
        missing_parent = self.root / "unmounted_drive" / "Strata-cache"
        cache = DiskSessionCache.from_config(str(missing_parent), 64, refused_type=FakeSessionRefused)
        self.assertFalse(cache.enabled)
        self.assertFalse(missing_parent.parent.exists())

        if os.geteuid() != 0:
            stub = self.root / "readonly_mount_stub"
            stub.mkdir(mode=0o555)
            try:
                ro_cache = DiskSessionCache.from_config(str(stub / "Strata-cache"), 64, refused_type=FakeSessionRefused)
                self.assertFalse(ro_cache.enabled)
                self.assertFalse((stub / "Strata-cache").exists())
            finally:
                stub.chmod(0o700)

    def test_consecutive_same_conversation_turns_do_zero_disk_io_until_switch_or_idle(self):
        svc = FakeService()
        cache_dir = self.root / "cache"
        cache = DiskSessionCache.from_config(str(cache_dir), 1, refused_type=FakeSessionRefused, idle_save_s=300.0)
        self.addCleanup(cache.close, svc)
        self.assertTrue(cache.enabled)

        conv_a_t1 = [100 + (i % 50) for i in range(1500)]
        gen_a_t1 = [200 + i for i in range(500)]  # full A after t1: 2000 tokens
        self.assertIsNone(svc.run_turn(cache, conv_a_t1, gen_a_t1))
        self.assertEqual(svc.engine.calls, [])

        # Turn 2 extends Conv A directly in engine RAM -> zero disk SAVE or RESTORE!
        conv_a_t2 = conv_a_t1 + gen_a_t1 + [300 + i for i in range(200)]
        gen_a_t2 = [400 + i for i in range(400)]  # full A after t2: 2600 tokens
        self.assertIsNone(svc.run_turn(cache, conv_a_t2, gen_a_t2))
        self.assertEqual(svc.engine.calls, [])

        # Turn 3 switches to completely different Conv B -> lazily saves Conv A (2600 tok) before Conv B runs
        conv_b_t1 = [900 + (i % 40) for i in range(1400)]
        gen_b_t1 = [950 + i for i in range(400)]  # full B after t1: 1800 tokens
        self.assertIsNone(svc.run_turn(cache, conv_b_t1, gen_b_t1))
        self.assertEqual([a for a, _ in svc.engine.calls], ["save"])

        # Turn 4 switches back to Conv A (LCP=2600 >= 1024) -> saves Conv B (1800 tok) AND restores Conv A!
        conv_a_t3 = conv_a_t2 + gen_a_t2 + [500 + i for i in range(200)]
        gen_a_t3 = [600 + i for i in range(400)]
        restored = svc.run_turn(cache, conv_a_t3, gen_a_t3)
        self.assertIsNotNone(restored)
        self.assertEqual(restored["tokens"], 2600)
        self.assertEqual([a for a, _ in svc.engine.calls], ["save", "save", "restore"])

    def test_prefix_superseding_replaces_shorter_snapshot_of_same_conversation(self):
        svc = FakeService()
        cache_dir = self.root / "cache"
        cache = DiskSessionCache.from_config(str(cache_dir), 1, refused_type=FakeSessionRefused, idle_save_s=300.0)
        self.addCleanup(cache.close, svc)

        conv_a = [10 + i for i in range(1200)]
        svc.run_turn(cache, conv_a, [2000 + i for i in range(400)])  # 1600 tokens
        # Switch to B -> saves A (1600 tokens)
        conv_b = [50000 + i for i in range(1200)]
        svc.run_turn(cache, conv_b, [60000 + i for i in range(400)])
        bins_after_first = sorted(p.name for p in cache_dir.glob("session-*.bin"))
        self.assertEqual(len(bins_after_first), 1)

        # Resume A and extend by +800 tokens (2400 tokens total), then close (which saves A and supersedes 1600-tok A)
        conv_a_ext = conv_a + [2000 + i for i in range(400)] + [3000 + i for i in range(200)]
        svc.run_turn(cache, conv_a_ext, [4000 + i for i in range(600)])  # 2400 tokens
        cache.close(svc)

        idx = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
        lengths = sorted(len(e["tokens"]) for e in idx["entries"])
        # Conv B (1600) + superseded Conv A (2400) — old 1600-tok Conv A bin was removed!
        self.assertEqual(lengths, [1600, 2400])
        self.assertNotIn(bins_after_first[0], {p.name for p in cache_dir.glob("session-*.bin")})
        self.assertEqual(len(list(cache_dir.glob("session-*.bin"))), 2)

    def test_lru_budget_evicts_oldest_on_save_and_on_startup_shrink(self):
        # 1024 bytes/token * 2048 tokens = 2 MiB per session. Budget = 5 MiB -> max 2 sessions fit.
        svc = FakeService(FakeEngine(bytes_per_token=1024))
        cache_dir = self.root / "cache"
        five_mib_in_gib = 5 / 1024
        cache = DiskSessionCache.from_config(
            str(cache_dir), five_mib_in_gib, refused_type=FakeSessionRefused, idle_save_s=300.0
        )

        for cid in range(3):
            prompt = [cid * 100000 + i for i in range(1500)]
            gen = [cid * 100000 + 50000 + i for i in range(548)]  # 2048 tokens = ~2 MiB
            svc.run_turn(cache, prompt, gen)
            time.sleep(0.01)
        cache.close(svc)

        idx = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
        # Oldest (cid=0) must have been evicted to stay <= 5 MiB; cid=1 and cid=2 remain
        self.assertEqual(len(idx["entries"]), 2)
        self.assertEqual([e["tokens"][0] for e in idx["entries"]], [100000, 200000])
        self.assertLessEqual(sum(e["bytes"] for e in idx["entries"]), cache.budget_bytes)

        # Now reopen with a smaller budget (2.5 MiB -> max 1 session): startup must LRU-trim immediately!
        shrunk = DiskSessionCache.from_config(
            str(cache_dir), 2.5 / 1024, refused_type=FakeSessionRefused, idle_save_s=300.0
        )
        self.addCleanup(shrunk.close, svc)
        idx2 = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(len(idx2["entries"]), 1)
        self.assertEqual(idx2["entries"][0]["tokens"][0], 200000)
        self.assertEqual(len(list(cache_dir.glob("session-*.bin"))), 1)

    def test_oversized_single_session_exceeding_budget_is_not_retained(self):
        # 2048 tokens * 1024 B = 2 MiB session, but budget is 1 MiB
        svc = FakeService(FakeEngine(bytes_per_token=1024))
        cache_dir = self.root / "cache"
        cache = DiskSessionCache.from_config(
            str(cache_dir), 1 / 1024, refused_type=FakeSessionRefused, idle_save_s=300.0
        )
        svc.run_turn(cache, [1 + i for i in range(1500)], [2000 + i for i in range(548)])
        cache.close(svc)
        self.assertEqual(list(cache_dir.glob("session-*.bin")), [])

    def test_engine_restart_invalidates_live_tokens_and_restores_from_disk(self):
        svc = FakeService()
        cache_dir = self.root / "cache"
        cache = DiskSessionCache.from_config(str(cache_dir), 1, refused_type=FakeSessionRefused, idle_save_s=0.05)
        self.addCleanup(cache.close, svc)

        conv_a = [700 + i for i in range(1300)]
        gen_a = [2000 + i for i in range(400)]
        svc.run_turn(cache, conv_a, gen_a)
        # Wait for idle save to flush Conv A to disk
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not list(cache_dir.glob("session-*.bin")):
            time.sleep(0.02)
        self.assertEqual(len(list(cache_dir.glob("session-*.bin"))), 1)

        # Simulate engine crash + restart (gen increments, engine RAM is now empty)
        svc.engine.gen += 1
        svc.engine.held_tokens = []

        # Next turn of the SAME conversation must detect gen change and restore from disk!
        restored = svc.run_turn(cache, conv_a + gen_a + [99, 100], [101, 102])
        self.assertIsNotNone(restored)
        self.assertEqual(restored["tokens"], 1700)

    def test_corrupt_index_and_orphan_bins_recover_on_startup(self):
        cache_dir = self.root / "cache"
        cache_dir.mkdir(mode=0o700)
        (cache_dir / "index.json").write_text("{corrupt-json", encoding="utf-8")
        orphan = cache_dir / "session-deadbeef0000.bin"
        orphan.write_bytes(b"orphan")
        stale_tmp = cache_dir / ".index-123.tmp"
        stale_tmp.write_bytes(b"partial")

        cache = DiskSessionCache.from_config(str(cache_dir), 1, refused_type=FakeSessionRefused, idle_save_s=300.0)
        self.assertTrue(cache.enabled)
        self.assertFalse(orphan.exists())
        self.assertFalse(stale_tmp.exists())

    def test_stale_or_refused_restore_evicts_entry_and_falls_back_to_next_or_cold_prefill(self):
        svc = FakeService()
        cache_dir = self.root / "cache"
        cache = DiskSessionCache.from_config(str(cache_dir), 1, refused_type=FakeSessionRefused, idle_save_s=300.0)

        conv_a = [42 + i for i in range(1300)]
        svc.run_turn(cache, conv_a, [5000 + i for i in range(300)])
        cache.close(svc)

        bins = list(cache_dir.glob("session-*.bin"))
        self.assertEqual(len(bins), 1)
        bad_path = str(bins[0])

        # Reopen in a fresh server where engine rejects the stale session with kind="invalid"
        svc2 = FakeService()
        svc2.engine.fail_restore_paths[bad_path] = FakeSessionRefused("SERR invalid model mismatch", kind="invalid")
        cache2 = DiskSessionCache.from_config(str(cache_dir), 1, refused_type=FakeSessionRefused, idle_save_s=300.0)
        self.addCleanup(cache2.close, svc2)

        # Request still succeeds (returns None -> cold prefill) and evicts the invalid .bin!
        restored = svc2.run_turn(cache2, conv_a + [5000 + i for i in range(300)] + [1], [2, 3])
        self.assertIsNone(restored)
        self.assertFalse(Path(bad_path).exists())

        # Now test transient RAM refusal (kind="memory"): request succeeds AND .bin is KEPT on disk!
        cache2.close(svc2)
        kept_bins = list(cache_dir.glob("session-*.bin"))
        self.assertEqual(len(kept_bins), 1)
        mem_path = str(kept_bins[0])

        svc3 = FakeService()
        svc3.engine.fail_restore_paths[mem_path] = FakeSessionRefused("not enough RAM", kind="memory")
        cache3 = DiskSessionCache.from_config(str(cache_dir), 1, refused_type=FakeSessionRefused, idle_save_s=300.0)
        self.addCleanup(cache3.close, svc3)
        self.assertIsNone(svc3.run_turn(cache3, [999999 + i for i in range(20)], [1]))
        self.assertIsNone(svc3.run_turn(cache3, conv_a + [5000 + i for i in range(300)] + [1, 2, 3, 4], [5]))
        self.assertTrue(Path(mem_path).exists())

    def test_failed_or_cancelled_turn_invalidates_live_state_without_saving_partial_tokens(self):
        svc = FakeService()
        cache_dir = self.root / "cache"
        cache = DiskSessionCache.from_config(str(cache_dir), 1, refused_type=FakeSessionRefused, idle_save_s=300.0)

        # Cancelled mid-read turn (ok=False) must NOT be saved when closing or switching
        svc.run_turn(cache, [1 + i for i in range(2000)], [99], ok=False)
        cache.close(svc)
        self.assertEqual(list(cache_dir.glob("session-*.bin")), [])

    def test_idle_timer_and_close_flush_dirty_conversation(self):
        svc = FakeService()
        cache_dir = self.root / "cache"
        cache = DiskSessionCache.from_config(str(cache_dir), 1, refused_type=FakeSessionRefused, idle_save_s=300.0)

        # Periodic 16,384-token checkpoint saves immediately at end of turn
        big_prompt = [1 + i for i in range(16000)]
        big_gen = [20000 + i for i in range(600)]  # 16,600 unsaved tokens >= 16,384
        svc.run_turn(cache, big_prompt, big_gen)
        self.assertEqual([a for a, _ in svc.engine.calls], ["save"])

        # Extending by +700 tokens (>= 512) marks dirty, and on_unload_locked flushes before unload!
        ext_prompt = big_prompt + big_gen + [30000 + i for i in range(300)]
        svc.run_turn(cache, ext_prompt, [40000 + i for i in range(400)])
        self.assertEqual([a for a, _ in svc.engine.calls], ["save"])
        with svc.fifo:
            cache.on_unload_locked(svc)
        self.assertEqual([a for a, _ in svc.engine.calls], ["save", "save"])

        # Storage refusal ("ENOSPC") evicts oldest entry and retries save once
        svc.run_turn(cache, [50000 + i for i in range(1500)], [60000 + i for i in range(200)])
        svc.engine.fail_next_save = FakeSessionRefused("disk full", kind="storage")
        cache.close(svc)
        self.assertEqual([a for a, _ in svc.engine.calls], ["save", "save", "save", "save"])

    def test_reasoning_think_stripping_no_false_switch_no_every_turn_checkpoint_and_supersedes(self):
        svc = FakeService()
        cache_dir = self.root / "think-cache"
        cache = DiskSessionCache.from_config(str(cache_dir), 1, refused_type=FakeSessionRefused, idle_save_s=300.0)
        self.addCleanup(cache.close, svc)

        def run_reasoning_turn(history_prefix: list[int], user_tokens: list[int], think_tokens: list[int], ans_tokens: list[int]):
            # Prompt ends with `<think>\n` (token 999901, 999902); next turn strips `<think>` and `\n` becomes `\n\n` (999903)
            prompt = history_prefix + user_tokens + [999901, 999902]
            full = prompt + think_tokens + ans_tokens
            with svc.fifo:
                cache.before_generate(svc, prompt)
                svc.engine.held_tokens = list(full)
                cache.after_generate(svc, full, ok=True, prompt_len=len(prompt))
            next_history = history_prefix + user_tokens + [999901, 999903] + ans_tokens
            return next_history

        # Turn 1: 2,000 user + 2,000 <think> + 300 answer -> no save yet
        hist = run_reasoning_turn([], [100 + i for i in range(2000)], [70000 + i for i in range(2000)], [80000 + i for i in range(300)])
        self.assertEqual(svc.engine.calls, [])

        # Turn 2 (same conversation, <think>_1 stripped!): MUST NOT trigger a false "switch" save!
        hist = run_reasoning_turn(hist, [200 + i for i in range(2000)], [72000 + i for i in range(2000)], [81000 + i for i in range(300)])
        self.assertEqual(svc.engine.calls, [])

        # Grow past 16,384 tokens -> triggers exactly ONE periodic checkpoint save
        hist = run_reasoning_turn(hist, [300 + i for i in range(11000)], [74000 + i for i in range(2000)], [82000 + i for i in range(300)])
        self.assertEqual([a for a, _ in svc.engine.calls], ["save"])
        self.assertEqual(len(list(cache_dir.glob("session-*.bin"))), 1)

        # Next turn (+800 user tokens, <think> stripped again): MUST NOT trigger another checkpoint save or switch save!
        hist = run_reasoning_turn(hist, [400 + i for i in range(800)], [76000 + i for i in range(2000)], [83000 + i for i in range(300)])
        self.assertEqual([a for a, _ in svc.engine.calls], ["save"])

        # Simulate cold engine restart -> restores checkpoint and runs 1 small turn WITHOUT re-saving immediately
        svc.engine.gen += 1
        svc.engine.held_tokens = []
        hist = run_reasoning_turn(hist, [500 + i for i in range(600)], [78000 + i for i in range(2000)], [84000 + i for i in range(300)])
        self.assertEqual([a for a, _ in svc.engine.calls], ["save", "restore"])

        # Flush updated conversation on unload -> MUST supersede the older snapshot (still only 1 .bin on disk!)
        with svc.fifo:
            cache.on_unload_locked(svc)
        self.assertEqual([a for a, _ in svc.engine.calls], ["save", "restore", "save"])
        self.assertEqual(len(cache.entries), 1)
        self.assertEqual(len(list(cache_dir.glob("session-*.bin"))), 1)

    def test_oversized_session_above_16k_does_not_rewrite_on_every_subsequent_turn(self):
        # 16,500 tokens * 64 bytes = ~1.0 MiB > 0.0005 GiB (~536 KiB) budget
        svc = FakeService(FakeEngine(bytes_per_token=64))
        cache_dir = self.root / "tiny-budget-cache"
        cache = DiskSessionCache.from_config(str(cache_dir), 0.0005, refused_type=FakeSessionRefused, idle_save_s=300.0)
        self.addCleanup(cache.close, svc)

        p1 = [1 + i for i in range(16000)]
        g1 = [20000 + i for i in range(600)]
        svc.run_turn(cache, p1, g1)
        self.assertEqual([a for a, _ in svc.engine.calls], ["save"])
        self.assertEqual(cache.entries, [])

        # Small follow-up turn (+600 tokens < 16,384) must NOT write & unlink another oversized .bin!
        p2 = p1 + g1 + [30000 + i for i in range(400)]
        svc.run_turn(cache, p2, [40000 + i for i in range(200)])
        self.assertEqual([a for a, _ in svc.engine.calls], ["save"])

    def test_close_waits_for_inflight_fifo_turn_and_malformed_index_entry_skipped(self):
        svc = FakeService()
        cache_dir = self.root / "close-race-cache"
        cache = DiskSessionCache.from_config(str(cache_dir), 1, refused_type=FakeSessionRefused, idle_save_s=300.0)
        cache._remember_service(svc)

        # Acquire fifo before live_dirty becomes True, simulating an in-flight turn during shutdown
        entered = threading.Event()

        def inflight_turn():
            with svc.fifo:
                entered.set()
                time.sleep(0.08)
                tokens = [10 + i for i in range(1500)]
                svc.engine.held_tokens = list(tokens)
                cache.after_generate(svc, tokens, ok=True)

        t = threading.Thread(target=inflight_turn)
        t.start()
        self.assertTrue(entered.wait(timeout=2.0))
        cache.close(svc)
        t.join(timeout=2.0)
        self.assertEqual([a for a, _ in svc.engine.calls], ["save"])
        self.assertEqual(len(cache.entries), 1)

        # Inject a malformed entry (non-int token) alongside the valid entry in index.json -> boot must skip bad entry, not crash!
        good_entry = dict(cache.entries[0])
        bad_bin = cache_dir / "session-bad123.bin"
        bad_bin.write_bytes(b"corrupt")
        (cache_dir / "index.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "entries": [
                        {"filename": "session-bad123.bin", "tokens": ["not_an_int"], "bytes": 7},
                        good_entry,
                    ],
                }
            ),
            encoding="utf-8",
        )
        reloaded = DiskSessionCache.from_config(str(cache_dir), 1, refused_type=FakeSessionRefused, idle_save_s=300.0)
        self.assertEqual(len(reloaded.entries), 1)
        self.assertEqual(reloaded.entries[0]["filename"], good_entry["filename"])
        self.assertFalse(bad_bin.exists())


class StrataControlIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="strata-dc-int-")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_runconfig_exposes_disk_cache_gib_and_protects_disk_cache_path(self):
        from serve import runconfig

        cfg = {"args": [], "disk_cache_path": "/mnt/cache", "disk_cache_gib": 64}
        self.assertEqual(runconfig.check("disk_cache_gib", 64, cfg), 64)
        self.assertEqual(runconfig.check("disk_cache_gib", 0, cfg), 0)
        self.assertEqual(runconfig.check("disk_cache_gib", 32.5, cfg), 32.5)
        with self.assertRaises(ValueError):
            runconfig.check("disk_cache_gib", -1, cfg)
        with self.assertRaises(ValueError):
            runconfig.check("disk_cache_path", "/other/path", cfg)

        updated, changed = runconfig.apply(cfg, {"disk_cache_gib": 48})
        self.assertEqual(changed, ["disk_cache_gib"])
        self.assertEqual(updated["disk_cache_gib"], 48)
        self.assertEqual(updated["disk_cache_path"], "/mnt/cache")

    def test_runner_derive_config_preserves_disk_cache_settings(self):
        from strata_cli.launch import runner, settings as S
        from strata_control import tune_knobs

        base = {"exe": "component:strata-engine", "args": ["--max-context", "262144"]}
        prev = {
            "args": ["--max-context", "262144"],
            "disk_cache_path": "/mnt/ssd/Strata-cache",
            "disk_cache_gib": 64,
        }
        ls = S.LaunchSettings(family="gyro", model="Gyro-M", access="lan", port=18931)
        derived = runner.derive_config(base, ls, bool_flags=tune_knobs.BOOL_FLAGS, prev=prev)
        self.assertEqual(derived.get("disk_cache_path"), "/mnt/ssd/Strata-cache")
        self.assertEqual(derived.get("disk_cache_gib"), 64)

    def test_service_end_to_end_turn_switch_unload_and_multimodal_bypass(self):
        from serve.frontend import ChatTemplate
        from serve import server as srv

        tok = srv.ByteTokenizer()

        class SessionMockEngine(srv.MockEngine):
            def __init__(self):
                super().__init__(tok, ["ok"], max_context=65536)
                self.gen = 1
                self.batch = 0
                self.unloaded = False
                self.session_save_reclaim = False
                self.held_tokens: list[int] = []
                self.calls: list[tuple[str, str]] = []

            def alive(self) -> bool:
                return not self.unloaded

            def unload(self) -> None:
                self.unloaded = True
                self.held_tokens = []

            def restart(self) -> None:
                self.unloaded = False
                self.gen += 1
                self.held_tokens = []

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                out = list(super().generate(ids, max_new, sampling, cancel, embeddings=embeddings))
                self.held_tokens = list(ids) + out
                for t in out:
                    yield t

            def session_file(self, action: str, path: str) -> dict:
                self.calls.append((action, path))
                if action == "save":
                    blob = json.dumps({"tokens": self.held_tokens}).encode("utf-8")
                    Path(path).write_bytes(blob)
                    return {"tokens": len(self.held_tokens), "bytes": len(blob), "ms": 2.0}
                if action == "restore":
                    data = json.loads(Path(path).read_bytes().decode("utf-8"))
                    self.held_tokens = list(data["tokens"])
                    return {"tokens": len(self.held_tokens), "bytes": 128, "ms": 1.5}
                raise ValueError(action)

        engine = SessionMockEngine()
        tpl_path = Path(srv.__file__).resolve().parent / "chat_template.jinja"
        svc = srv.Service(engine, tok, ChatTemplate(tpl_path))
        cache_dir = self.root / "svc-cache"
        svc.disk_cache = DiskSessionCache.from_config(
            str(cache_dir), 1, refused_type=srv.SessionRefused, idle_save_s=300.0
        )
        self.addCleanup(svc.disk_cache.close, svc)
        self.assertTrue(svc.disk_cache.enabled)

        def run_ids(prompt_ids: list[int], *, with_emb: bool = False):
            cancel = threading.Event()
            if with_emb:
                emb_file = self.root / "dummy.sve"
                emb_file.write_bytes(b"emb")
                svc.embeddings.path = str(emb_file)
            events = list(svc.run(prompt_ids, False, None, 16, {}, cancel))
            self.assertEqual(events[-1][0], "done")
            return events[-1][1]

        # Turn 1 + Turn 2 of Conv A: zero disk I/O during active streak
        conv_a1 = [65 + (i % 26) for i in range(1400)]
        run_ids(conv_a1)
        self.assertEqual(engine.calls, [])
        conv_a2 = list(engine.held_tokens) + [90] * 600
        run_ids(conv_a2)
        self.assertEqual(engine.calls, [])
        saved_a_tokens = list(engine.held_tokens)

        # Switch to Conv B -> lazily saves Conv A
        conv_b1 = [48 + (i % 10) for i in range(1500)]
        run_ids(conv_b1)
        self.assertEqual([a for a, _ in engine.calls], ["save"])
        saved_b_tokens = list(engine.held_tokens)

        # Multimodal request with embeddings flushes dirty Conv B BEFORE overwriting the engine slot,
        # and invalidates live_tokens so resuming Conv B restores Conv B's genuine KV tokens (not [70]*1500)!
        run_ids([70] * 1500, with_emb=True)
        self.assertEqual([a for a, _ in engine.calls], ["save", "save"])
        self.assertEqual(svc.disk_cache.live_tokens, [])
        conv_b_saved_blob = json.loads(Path(engine.calls[-1][1]).read_bytes().decode("utf-8"))
        self.assertEqual(conv_b_saved_blob["tokens"], saved_b_tokens)

        # Resuming Conv B right after multimodal turn restores Conv B from disk!
        conv_b2 = saved_b_tokens + [93] * 100
        run_ids(conv_b2)
        self.assertEqual([a for a, _ in engine.calls], ["save", "save", "restore"])

        # Switch back to Conv A -> restores Conv A (Conv B grew only +102 < 512 tokens so not dirty)
        conv_a3 = saved_a_tokens + [91] * 600
        run_ids(conv_a3)
        self.assertEqual([a for a, _ in engine.calls], ["save", "save", "restore", "restore"])

        # Unload saves dirty Conv A before stopping engine; next turn reloads engine AND restores Conv A!
        self.assertEqual(svc.unload(), "unloaded")
        self.assertEqual([a for a, _ in engine.calls], ["save", "save", "restore", "restore", "save"])
        conv_a4 = list(svc.disk_cache.entries[-1]["tokens"]) + [92] * 100
        run_ids(conv_a4)
        self.assertEqual([a for a, _ in engine.calls], ["save", "save", "restore", "restore", "save", "restore"])

        # Manual /slots/0?action=restore invalidates disk_cache live_tokens
        slot_dir = self.root / "slots"
        slot_dir.mkdir()
        (slot_dir / "manual.bin").write_bytes(json.dumps({"tokens": [1, 2, 3]}).encode("utf-8"))
        svc.slot_save_path = str(slot_dir)
        code, _ = svc.slot_action("0", "restore", "manual.bin")
        self.assertEqual(code, 200)
        self.assertEqual(svc.disk_cache.live_tokens, [])

    def test_http_v1_chat_completions_persists_and_restores_across_server_restart(self):
        import urllib.request
        from serve.frontend import ChatTemplate
        from serve import server as srv

        tok = srv.ByteTokenizer()
        tpl = ChatTemplate(Path(srv.__file__).resolve().parent / "chat_template.jinja")
        cache_dir = self.root / "http-cache"

        class HttpSessionMockEngine(srv.MockEngine):
            def __init__(self):
                super().__init__(tok, ["Thinking.</think>\n\nassistant reply"], max_context=65536)
                self.gen = 1
                self.batch = 0
                self.session_save_reclaim = False
                self.held_tokens: list[int] = []
                self.calls: list[tuple[str, str]] = []

            def alive(self) -> bool:
                return True

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                out = list(super().generate(ids, max_new, sampling, cancel, embeddings=embeddings))
                # Exclude trailing EOS from KV snapshot, matching strata-engine
                self.held_tokens = list(ids) + out[:-1]
                for t in out:
                    yield t

            def session_file(self, action: str, path: str) -> dict:
                self.calls.append((action, path))
                if action == "save":
                    blob = json.dumps({"tokens": self.held_tokens}).encode("utf-8")
                    Path(path).write_bytes(blob)
                    return {"tokens": len(self.held_tokens), "bytes": len(blob), "ms": 2.5}
                if action == "restore":
                    data = json.loads(Path(path).read_bytes().decode("utf-8"))
                    self.held_tokens = list(data["tokens"])
                    return {"tokens": len(self.held_tokens), "bytes": 128, "ms": 1.8}
                raise ValueError(action)

        def post_chat(port: int, messages: list[dict], **extra) -> dict:
            body = json.dumps({"model": "test", "messages": messages, "stream": False, "max_tokens": 64, **extra}).encode("utf-8")
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                self.assertEqual(resp.status, 200)
                return json.loads(resp.read().decode("utf-8"))

        # --- Server boot #1 ---
        eng1 = HttpSessionMockEngine()
        svc1 = srv.Service(eng1, tok, tpl)
        svc1.disk_cache = DiskSessionCache.from_config(
            str(cache_dir), 64, refused_type=srv.SessionRefused, idle_save_s=300.0
        )
        httpd1 = srv.serve(svc1, host="127.0.0.1", port=0)
        port1 = httpd1.server_address[1]
        try:
            long_doc_a = "Alpha context sentence. " * 80  # ~1920 byte tokens (> 1024)
            msgs_a = [{"role": "user", "content": long_doc_a + "Question 1?"}]
            ans1 = post_chat(port1, msgs_a)
            reply1 = ans1["choices"][0]["message"]["content"]
            self.assertIn("assistant reply", reply1)
            self.assertEqual(eng1.calls, [])
        finally:
            httpd1.shutdown()
            httpd1.server_close()
            svc1.disk_cache.close(svc1)

        # Clean shutdown flushed Conversation A to disk (index.json + 1 .bin file)
        self.assertEqual([a for a, _ in eng1.calls], ["save"])
        self.assertTrue((cache_dir / "index.json").is_file())
        self.assertEqual(len(list(cache_dir.glob("session-*.bin"))), 1)

        # --- Server boot #2 (simulating reboot) ---
        eng2 = HttpSessionMockEngine()
        svc2 = srv.Service(eng2, tok, tpl)
        svc2.disk_cache = DiskSessionCache.from_config(
            str(cache_dir), 64, refused_type=srv.SessionRefused, idle_save_s=300.0
        )
        self.assertEqual(len(svc2.disk_cache.entries), 1)
        httpd2 = srv.serve(svc2, host="127.0.0.1", port=0)
        port2 = httpd2.server_address[1]
        try:
            msgs_a_turn2 = msgs_a + [
                {"role": "assistant", "content": reply1},
                {"role": "user", "content": "Follow-up question 2?"},
            ]
            ans2 = post_chat(port2, msgs_a_turn2)
            self.assertIn("assistant reply", ans2["choices"][0]["message"]["content"])
            # Boot #2 restored Conversation A from disk before generating!
            self.assertEqual([a for a, _ in eng2.calls], ["restore"])

            # One-shot calls with strata_checkpoint=False (skyrat) or cache_prompt=False
            # never write their own sessions to disk, even when > 1024 tokens:
            skyrat_doc = "Skyrat batch frame prompt. " * 80
            for i in range(3):
                post_chat(
                    port2,
                    [{"role": "user", "content": f"{skyrat_doc} frame {i}"}],
                    strata_checkpoint=False,
                )
            post_chat(
                port2,
                [{"role": "user", "content": f"{skyrat_doc} uncached probe"}],
                cache_prompt=False,
            )
            self.assertEqual([a for a, _ in eng2.calls], ["restore"])
        finally:
            httpd2.shutdown()
            httpd2.server_close()
            svc2.disk_cache.close(svc2)

        # Shutdown after skyrat one-shot calls wrote ZERO extra .bin files
        self.assertEqual([a for a, _ in eng2.calls], ["restore"])
        self.assertEqual(len(list(cache_dir.glob("session-*.bin"))), 1)

        # --- Out-of-the-box blank disk_cache_path="" check ---
        svc3 = srv.Service(HttpSessionMockEngine(), tok, tpl)
        svc3.disk_cache = DiskSessionCache.from_config("", 64, refused_type=srv.SessionRefused)
        self.assertFalse(svc3.disk_cache.enabled)
        httpd3 = srv.serve(svc3, host="127.0.0.1", port=0)
        try:
            ans3 = post_chat(httpd3.server_address[1], [{"role": "user", "content": "Short hello"}])
            self.assertIn("assistant reply", ans3["choices"][0]["message"]["content"])
        finally:
            httpd3.shutdown()
            httpd3.server_close()
            svc3.disk_cache.close(svc3)

    def test_fatal_restore_evicts_poison_bin_restarts_engine_and_completes_turn(self):
        from serve.frontend import ChatTemplate
        from serve import server as srv

        tok = srv.ByteTokenizer()

        class FatalRestoreEngine(srv.MockEngine):
            def __init__(self):
                super().__init__(tok, ["ok"], max_context=65536)
                self.gen = 1
                self.batch = 0
                self.ended = False
                self.unloaded = False
                self.session_save_reclaim = False
                self.held_tokens: list[int] = []
                self.calls: list[tuple[str, str]] = []
                self.poison_next_restore = False

            def alive(self) -> bool:
                return not self.ended and not self.unloaded

            def restart(self) -> None:
                self.ended = False
                self.unloaded = False
                self.gen += 1
                self.held_tokens = []

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                if not self.alive():
                    raise srv.EngineDied("engine is dead during generate()")
                out = list(super().generate(ids, max_new, sampling, cancel, embeddings=embeddings))
                self.held_tokens = list(ids) + out[:-1]
                for t in out:
                    yield t

            def session_file(self, action: str, path: str) -> dict:
                self.calls.append((action, path))
                if action == "save":
                    blob = json.dumps({"tokens": self.held_tokens}).encode("utf-8")
                    Path(path).write_bytes(blob)
                    return {"tokens": len(self.held_tokens), "bytes": len(blob), "ms": 1.0}
                if action == "restore":
                    if self.poison_next_restore:
                        self.poison_next_restore = False
                        self.ended = True
                        raise srv.EngineDied("FATAL mid-file corruption during session restore")
                    data = json.loads(Path(path).read_bytes().decode("utf-8"))
                    self.held_tokens = list(data["tokens"])
                    return {"tokens": len(self.held_tokens), "bytes": 128, "ms": 1.0}
                raise ValueError(action)

        engine = FatalRestoreEngine()
        svc = srv.Service(engine, tok, ChatTemplate(Path(srv.__file__).resolve().parent / "chat_template.jinja"))
        cache_dir = self.root / "fatal-cache"
        svc.disk_cache = DiskSessionCache.from_config(
            str(cache_dir), 1, refused_type=srv.SessionRefused, idle_save_s=300.0
        )
        self.addCleanup(svc.disk_cache.close, svc)

        conv_a = [65 + (i % 26) for i in range(1400)]
        events1 = list(svc.run(conv_a, False, None, 16, {}, threading.Event()))
        self.assertEqual(events1[-1][0], "done")
        saved_a = list(engine.held_tokens)
        svc.disk_cache.close(svc)

        # Re-open cache as fresh boot and poison the saved .bin restore with EngineDied("FATAL ...")
        engine2 = FatalRestoreEngine()
        engine2.poison_next_restore = True
        svc2 = srv.Service(engine2, tok, ChatTemplate(Path(srv.__file__).resolve().parent / "chat_template.jinja"))
        svc2.disk_cache = DiskSessionCache.from_config(
            str(cache_dir), 1, refused_type=srv.SessionRefused, idle_save_s=300.0
        )
        self.addCleanup(svc2.disk_cache.close, svc2)
        self.assertEqual(len(svc2.disk_cache.entries), 1)

        # Request MUST fail-open: evict poison .bin, restart engine, and complete generation cleanly!
        events2 = list(svc2.run(saved_a + [90] * 200, False, None, 16, {}, threading.Event()))
        self.assertEqual(events2[-1][0], "done")
        self.assertGreaterEqual(engine2.gen, 2)
        self.assertEqual(list(cache_dir.glob("session-*.bin")), [])

    def test_pre_generate_error_does_not_record_unexecuted_prompt_as_live(self):
        from serve.frontend import ChatTemplate
        from serve import server as srv

        tok = srv.ByteTokenizer()

        class AliveTextEngine(srv.MockEngine):
            def __init__(self):
                super().__init__(tok, ["ok"], max_context=65536)
                self.gen = 1
                self.batch = 0
                self.session_save_reclaim = False
                self.held_tokens: list[int] = []

            def alive(self) -> bool:
                return True

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                out = list(super().generate(ids, max_new, sampling, cancel, embeddings=embeddings))
                self.held_tokens = list(ids) + out[:-1]
                for t in out:
                    yield t

            def session_file(self, action: str, path: str) -> dict:
                return {"tokens": 0, "bytes": 0, "ms": 0.0}

        class BrokenVision:
            elastic = False

            def alive(self) -> bool:
                return False

            def restart(self) -> None:
                raise RuntimeError("mmproj missing")

        engine = AliveTextEngine()
        svc = srv.Service(engine, tok, ChatTemplate(Path(srv.__file__).resolve().parent / "chat_template.jinja"))
        svc.disk_cache = DiskSessionCache.from_config(
            str(self.root / "pregen-cache"), 1, refused_type=srv.SessionRefused, idle_save_s=300.0
        )
        self.addCleanup(svc.disk_cache.close, svc)

        # First turn succeeds cleanly without vision
        conv_a = [65 + (i % 26) for i in range(1400)]
        list(svc.run(conv_a, False, None, 16, {}, threading.Event()))
        prev_live = list(svc.disk_cache.live_tokens)
        self.assertGreaterEqual(len(prev_live), 1400)

        # Now attach broken non-elastic vision so ensure_loaded() raises EngineDied BEFORE engine.generate() runs
        svc.vision = BrokenVision()
        unexecuted_ids = [49] * 1600
        with self.assertRaises(srv.EngineDied):
            list(svc.run(unexecuted_ids, False, None, 16, {}, threading.Event()))
        self.assertNotEqual(svc.disk_cache.live_tokens, unexecuted_ids)

    @staticmethod
    def _make_session_bin(
        path: Path,
        *,
        version: int = 1,
        model_fp: int = 0x3596F3A865B63D1E,
        config_fp: int = 0x8341DE3075756AB1,
        payload: bytes | None = None,
        live_tokens: list[int] | None = None,
        checkpoint_lens: list[int] | None = None,
        corrupt_header_hash: bool = False,
        bad_trailer: bool = False,
    ) -> int:
        import struct
        from serve.disk_cache import session_hash64

        if payload is None:
            if live_tokens is not None or checkpoint_lens is not None:
                ltoks = list(live_tokens or [1] * 4000)
                ck_lens = list(checkpoint_lens or [])

                def pack_ckpt(ids_list: list[int]) -> bytes:
                    parts = [
                        struct.pack(f"<Q{len(ids_list)}i", len(ids_list), *ids_list),
                        struct.pack("<Q", 0),  # 0 images
                    ]
                    for _ in range(5):
                        parts.append(struct.pack("<Q", 8) + b"\x00" * 8)
                    parts.append(struct.pack("<Q", 1))  # used
                    return b"".join(parts)

                body = [
                    struct.pack("<18q", *([1] * 18)),  # geometry
                    struct.pack("<qqQ", 0, 48, 0),     # layer_lo, layer_hi, cvec
                    pack_ckpt(ltoks),
                    struct.pack("<Q", len(ck_lens)),
                ]
                for cl in ck_lens:
                    body.append(pack_ckpt(ltoks[:cl]))
                body.append(struct.pack("<Q", 0))      # 0 kv layers
                payload = b"".join(body)
            else:
                payload = b"\x00" * 128

        hdr_prefix = struct.pack(
            "<8sIIQQQQQ",
            b"STRSESS\x01",
            version,
            64,
            model_fp,
            config_fp,
            len(payload),
            0,
            0,
        )
        hh = session_hash64(hdr_prefix, 0)
        if corrupt_header_hash:
            hh ^= 1
        hdr = hdr_prefix + struct.pack("<Q", hh)
        ph = session_hash64(payload, 0)
        trailer = struct.pack("<Q8s", ph, b"BADEND!!\x00"[:8] if bad_trailer else b"STRSEND\x01")
        blob = hdr + payload + trailer
        path.write_bytes(blob)
        return len(blob)

    def test_prune_incompatible_removes_mismatched_and_corrupt_bins_and_keeps_matching(self):
        cache_dir = self.root / "prune-cache"
        cache_dir.mkdir()
        specs = [
            ("aaaa11111111", {"model_fp": 0x3596F3A865B63D1E, "config_fp": 0x8341DE3075756AB1}),
            ("bbbb22222222", {"model_fp": 0x3596F3A865B63D1E, "config_fp": 0xC24C8F80CC204E4C}),
            ("cccc33333333", {"model_fp": 0x1111111111111111, "config_fp": 0x8341DE3075756AB1}),
            ("dddd44444444", {"version": 2}),
            ("eeee55555555", {"corrupt_header_hash": True}),
            ("ffff66666666", {"bad_trailer": True}),
        ]
        entries = []
        for sid, kw in specs:
            fname = f"session-{sid}.bin"
            sz = self._make_session_bin(cache_dir / fname, **kw)
            entries.append(
                {
                    "id": sid,
                    "filename": fname,
                    "tokens": [10, 20, 30] * 500,
                    "prompt_len": 1500,
                    "bytes": sz,
                    "last_used": 1000.0,
                    "created_at": 1000.0,
                }
            )
        (cache_dir / "index.json").write_text(json.dumps({"version": 1, "entries": entries}), encoding="utf-8")

        cache = DiskSessionCache.from_config(str(cache_dir), 1)
        self.addCleanup(cache.close)
        self.assertEqual(len(cache.entries), 6)

        pruned = cache.prune_incompatible({"session_fp": "v1:3596f3a865b63d1e:8341de3075756ab1"})
        self.assertEqual(pruned, 5)
        self.assertEqual([e["id"] for e in cache.entries], ["aaaa11111111"])
        self.assertEqual([p.name for p in sorted(cache_dir.glob("session-*.bin"))], ["session-aaaa11111111.bin"])
        saved_idx = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
        self.assertEqual([e["id"] for e in saved_idx["entries"]], ["aaaa11111111"])
        self.assertEqual(cache.metrics_view()["sessions"], 1)

    def test_prune_incompatible_noop_when_engine_info_lacks_session_fp(self):
        cache_dir = self.root / "noop-prune"
        cache = DiskSessionCache.from_config(str(cache_dir), 1)
        self.addCleanup(cache.close)
        bpath = cache_dir / "session-123456789abc.bin"
        bpath.write_bytes(b"mock-bytes" * 20)
        cache.entries = [
            {
                "id": "123456789abc",
                "filename": "session-123456789abc.bin",
                "tokens": [1] * 1200,
                "prompt_len": 1200,
                "bytes": bpath.stat().st_size,
                "last_used": 1000.0,
                "created_at": 1000.0,
            }
        ]
        self.assertEqual(cache.prune_incompatible(None), 0)
        self.assertEqual(cache.prune_incompatible({}), 0)
        self.assertEqual(len(cache.entries), 1)
        self.assertTrue(bpath.is_file())

    def test_service_ensure_loaded_prunes_incompatible_sessions_on_reload(self):
        from serve.frontend import ChatTemplate
        from serve import server as srv

        tok = srv.ByteTokenizer()
        cache_dir = self.root / "reload-prune"
        cache_dir.mkdir()
        stale_tokens = [65 + (i % 26) for i in range(1400)]
        sz = self._make_session_bin(
            cache_dir / "session-deadbeef0001.bin",
            model_fp=0x3596F3A865B63D1E,
            config_fp=0xC24C8F80CC204E4C,  # stale 0.1.60 fingerprint
        )
        (cache_dir / "index.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "entries": [
                        {
                            "id": "deadbeef0001",
                            "filename": "session-deadbeef0001.bin",
                            "tokens": stale_tokens,
                            "prompt_len": len(stale_tokens),
                            "bytes": sz,
                            "last_used": 1000.0,
                            "created_at": 1000.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        class LazyFpEngine(srv.MockEngine):
            def __init__(self):
                super().__init__(tok, ["ok"], max_context=65536)
                self.gen = 0
                self.batch = 0
                self.unloaded = True
                self.session_save_reclaim = False
                self.info = {}
                self.restore_calls = 0

            def alive(self) -> bool:
                return not self.unloaded

            def restart(self) -> None:
                self.unloaded = False
                self.gen += 1
                self.info = {"session_fp": "v1:3596f3a865b63d1e:8341de3075756ab1"}

            def session_file(self, action: str, path: str) -> dict:
                if action == "restore":
                    self.restore_calls += 1
                return {"tokens": 0, "bytes": 0, "ms": 1.0}

        engine = LazyFpEngine()
        svc = srv.Service(engine, tok, ChatTemplate(Path(srv.__file__).resolve().parent / "chat_template.jinja"))
        svc.disk_cache = DiskSessionCache.from_config(
            str(cache_dir), 1, refused_type=srv.SessionRefused, idle_save_s=300.0
        )
        self.addCleanup(svc.disk_cache.close, svc)
        self.assertEqual(len(svc.disk_cache.entries), 1)

        events = list(svc.run(stale_tokens + [90] * 100, False, None, 16, {}, threading.Event()))
        self.assertEqual(events[-1][0], "done")
        self.assertEqual(engine.restore_calls, 0)
        self.assertEqual(svc.disk_cache.entries, [])
        self.assertEqual(list(cache_dir.glob("session-*.bin")), [])

    def test_reconcile_index_backfills_checkpoints_from_existing_bin_files(self):
        cache_dir = self.root / "backfill-ckpts"
        cache_dir.mkdir()
        toks = [10 + (i % 50) for i in range(3600)]
        sz = self._make_session_bin(
            cache_dir / "session-112233445566.bin",
            live_tokens=toks,
            checkpoint_lens=[512, 1536, 3400],
        )
        # Legacy index.json without "checkpoints" field:
        (cache_dir / "index.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "entries": [
                        {
                            "id": "112233445566",
                            "filename": "session-112233445566.bin",
                            "tokens": toks,
                            "prompt_len": 3405,
                            "bytes": sz,
                            "last_used": 1000.0,
                            "created_at": 1000.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        cache = DiskSessionCache.from_config(str(cache_dir), 1)
        self.addCleanup(cache.close)
        self.assertEqual(cache.entries[0].get("checkpoints"), [512, 1536, 3400])

    def test_checkpoint_aware_restore_skips_unreachable_lcp_and_reports_true_reuse(self):
        from types import SimpleNamespace

        # 1) Single-checkpoint .bin (ckpt only at 3800): prompt with LCP=2000 (< 3800) CANNOT resume -> must NOT restore!
        dir1 = self.root / "single-ckpt"
        dir1.mkdir()
        base_toks = [20 + (i % 40) for i in range(4000)]
        sz1 = self._make_session_bin(
            dir1 / "session-aaaaaaaa0001.bin",
            live_tokens=base_toks,
            checkpoint_lens=[3800],
        )
        (dir1 / "index.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "entries": [
                        {
                            "id": "aaaaaaaa0001",
                            "filename": "session-aaaaaaaa0001.bin",
                            "tokens": base_toks,
                            "prompt_len": 3805,
                            "bytes": sz1,
                            "last_used": 1000.0,
                            "created_at": 1000.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        cache1 = DiskSessionCache.from_config(str(dir1), 1)
        self.addCleanup(cache1.close)

        restore_calls: list[str] = []

        def fake_session_file(action: str, path: str) -> dict:
            if action == "restore":
                restore_calls.append(path)
                return {"tokens": 4000, "bytes": sz1, "ms": 12.0}
            raise ValueError(action)

        svc1 = SimpleNamespace(
            engine=SimpleNamespace(gen=1, batch=0, session_file=fake_session_file),
            fifo=threading.Lock(),
            status_lock=threading.Lock(),
            status={"busy": False, "queued": 0},
            loaded=lambda: True,
        )
        mid_edited_prompt = base_toks[:2000] + [999] * 2100
        res1 = cache1.before_generate(svc1, mid_edited_prompt)
        self.assertIsNone(res1)
        self.assertEqual(restore_calls, [])
        self.assertEqual(cache1.restores, 0)

        # 2) Multi-checkpoint .bin (ckpts at [800, 1600, 3800]): same LCP=2000 -> restores and reports exact reuse=1600!
        dir2 = self.root / "multi-ckpt"
        dir2.mkdir()
        sz2 = self._make_session_bin(
            dir2 / "session-bbbbbbbb0002.bin",
            live_tokens=base_toks,
            checkpoint_lens=[800, 1600, 3800],
        )
        (dir2 / "index.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "entries": [
                        {
                            "id": "bbbbbbbb0002",
                            "filename": "session-bbbbbbbb0002.bin",
                            "tokens": base_toks,
                            "prompt_len": 3805,
                            "bytes": sz2,
                            "last_used": 1000.0,
                            "created_at": 1000.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        cache2 = DiskSessionCache.from_config(str(dir2), 1)
        self.addCleanup(cache2.close)
        res2 = cache2.before_generate(svc1, mid_edited_prompt)
        self.assertIsNotNone(res2)
        self.assertEqual(len(restore_calls), 1)
        self.assertEqual(cache2.restores, 1)
        self.assertEqual(cache2.restored_tokens, 1600)
        self.assertEqual(cache2.metrics_view()["last_event"]["reused_tokens"], 1600)
        self.assertEqual(cache2.metrics_view()["last_event"]["tokens"], 4000)


if __name__ == "__main__":
    unittest.main()
