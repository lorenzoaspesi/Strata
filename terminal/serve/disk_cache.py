"""Automatic token-prefix-matched disk session cache for single-slot Strata serving.

Coordinates Strata's native `SAVE` / `RESTORE` engine commands inside `Service.fifo`:
- Zero disk I/O during consecutive turns of the active live conversation.
- Lazy `SAVE` when switching conversations, after idle timeout, before idle unload,
  at periodic token growth milestones, and on clean server shutdown.
- Automatic `RESTORE` when a cached session offers >= MIN_RESTORE_GAIN more prefix
  tokens than the engine's current live state.
- Reboot-persistent `index.json` + GiB LRU eviction.
"""
from __future__ import annotations

import json
import math
import os
import re
import secrets
import stat
import struct
import tempfile
import threading
import time
import weakref
from pathlib import Path
from typing import Sequence

MIN_RESTORE_GAIN = 1024
MIN_SAVE_TOKENS = 1024
MIN_SAVE_GAIN = 512
PERIODIC_SAVE_TOKENS = 16384
THINK_STRIP_SLACK = 16
IDLE_SAVE_S = 30.0
DEFAULT_DISK_CACHE_GIB = 64.0
CLOSE_WAIT_S = 30.0
SESSION_PREFIX = "session-"
SESSION_SUFFIX = ".bin"
INDEX_NAME = "index.json"

SESSION_MAGIC = b"STRSESS\x01"
SESSION_END_MAGIC = b"STRSEND\x01"
SESSION_HEADER_BYTES = 64
SESSION_TRAILER_BYTES = 16
_SESSION_FP_RE = re.compile(r"^v([0-9]+):([0-9a-fA-F]{16}):([0-9a-fA-F]{16})$")

_MASK64 = (1 << 64) - 1
_P1 = 0x9E3779B185EBCA87
_P2 = 0xC2B2AE3D27D4EB4F
_P3 = 0x165667B19E3779F9
_P4 = 0x85EBCA77C2B2AE63
_P5 = 0x27D4EB2F165667C5


def _rotl64(x: int, r: int) -> int:
    x &= _MASK64
    return ((x << r) | (x >> (64 - r))) & _MASK64


def _round1_64(acc: int, w: int) -> int:
    return (_rotl64((acc + ((w * _P2) & _MASK64)) & _MASK64, 31) * _P1) & _MASK64


def session_hash64(data: bytes | memoryview, seed: int = 0) -> int:
    """Exact port of `strata::core::session_hash64` (`src/core/conversation_file.cpp`)."""
    lane0 = (seed + _P1 + _P2) & _MASK64
    lane1 = (seed + _P2) & _MASK64
    lane2 = seed & _MASK64
    lane3 = (seed - _P1) & _MASK64
    mv = memoryview(data)
    n = len(mv)
    off = 0
    while off + 32 <= n:
        w0, w1, w2, w3 = struct.unpack_from("<4Q", mv, off)
        lane0 = _round1_64(lane0, w0)
        lane1 = _round1_64(lane1, w1)
        lane2 = _round1_64(lane2, w2)
        lane3 = _round1_64(lane3, w3)
        off += 32
    h = (_rotl64(lane0, 1) + _rotl64(lane1, 7) + _rotl64(lane2, 12) + _rotl64(lane3, 18)) & _MASK64
    for lv in (lane0, lane1, lane2, lane3):
        h ^= _round1_64(0, lv)
        h = (h * _P1 + _P4) & _MASK64
    h = (h + n) & _MASK64
    while off + 8 <= n:
        (w,) = struct.unpack_from("<Q", mv, off)
        h ^= _round1_64(0, w)
        h = (_rotl64(h, 27) * _P1 + _P4) & _MASK64
        off += 8
    while off < n:
        h ^= (mv[off] * _P5) & _MASK64
        h = (_rotl64(h, 11) * _P1) & _MASK64
        off += 1
    h ^= h >> 33
    h = (h * _P2) & _MASK64
    h ^= h >> 29
    h = (h * _P3) & _MASK64
    h ^= h >> 32
    return h


def _parse_session_fp(raw: object) -> tuple[int, int, int] | None:
    if not isinstance(raw, str):
        return None
    m = _SESSION_FP_RE.match(raw.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2), 16), int(m.group(3), 16)


def _session_bin_compatible(path: str | Path, want_ver: int, want_model: int, want_cfg: int) -> bool:
    try:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode) or st.st_size < SESSION_HEADER_BYTES + SESSION_TRAILER_BYTES:
            return False
        with open(path, "rb") as fh:
            hdr = fh.read(SESSION_HEADER_BYTES)
            if len(hdr) != SESSION_HEADER_BYTES:
                return False
            magic, ver, hsize, model_fp, cfg_fp, payload, r0, r1, hh = struct.unpack("<8sIIQQQQQQ", hdr)
            if (
                magic != SESSION_MAGIC
                or hsize != SESSION_HEADER_BYTES
                or r0 != 0
                or r1 != 0
                or SESSION_HEADER_BYTES + payload + SESSION_TRAILER_BYTES != st.st_size
                or ver != want_ver
                or model_fp != want_model
                or cfg_fp != want_cfg
                or session_hash64(hdr[:56], 0) != hh
            ):
                return False
            fh.seek(-SESSION_TRAILER_BYTES, os.SEEK_END)
            trl = fh.read(SESSION_TRAILER_BYTES)
            if len(trl) != SESSION_TRAILER_BYTES or trl[8:16] != SESSION_END_MAGIC:
                return False
        return True
    except OSError:
        return False


def lcp_len(a: Sequence[int], b: Sequence[int]) -> int:
    """Length of the longest common token prefix between `a` and `b`."""
    i = 0
    for x, y in zip(a, b):
        if x != y:
            break
        i += 1
    return i


def _is_network_or_invalid_path(raw: str) -> bool:
    if "\x00" in raw:
        return True
    norm = raw.replace("/", "\\")
    return norm.startswith("\\\\") or norm.startswith("\\??\\")


def _safe_unlink(path: str | Path) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


class DiskSessionCache:
    def __init__(
        self,
        *,
        enabled: bool,
        directory: str | None,
        budget_bytes: int,
        refused_type: type[BaseException] = Exception,
        idle_save_s: float = IDLE_SAVE_S,
    ) -> None:
        self.enabled = enabled
        self.dir = directory
        self.budget_bytes = budget_bytes
        self.refused_type = refused_type
        self.idle_save_s = max(0.01, float(idle_save_s))

        self.entries: list[dict] = []
        self.live_tokens: list[int] = []
        self.live_prompt_len: int = 0
        self.live_dirty: bool = False
        self.last_saved_len: int = 0
        self.live_gen: int = 0

        self._stats_lock = threading.Lock()
        self.saves: int = 0
        self.restores: int = 0
        self.evictions: int = 0
        self.restored_tokens: int = 0
        self.last_event: dict | None = None
        self._cached_view: dict = {}

        self._cv = threading.Condition()
        self._idle_deadline: float | None = None
        self._closing = False
        self._idle_thread: threading.Thread | None = None
        self._svc_ref: weakref.ReferenceType | None = None

        if self.enabled and self.dir is not None:
            self._load_and_reconcile_index()
        self._publish_view()

    @classmethod
    def from_config(
        cls,
        raw_path: str | None,
        raw_gib: float | int | None,
        *,
        cwd: str | None = None,
        refused_type: type[BaseException] = Exception,
        idle_save_s: float = IDLE_SAVE_S,
    ) -> "DiskSessionCache":
        if raw_gib is None:
            gib = DEFAULT_DISK_CACHE_GIB
        elif (
            isinstance(raw_gib, bool)
            or not isinstance(raw_gib, (int, float))
            or not math.isfinite(raw_gib)
            or raw_gib < 0
        ):
            raise ValueError(f"disk_cache_gib must be a finite number >= 0, not {raw_gib!r}")
        else:
            gib = float(raw_gib)

        if raw_path is None:
            return cls(enabled=False, directory=None, budget_bytes=0, refused_type=refused_type, idle_save_s=idle_save_s)
        if not isinstance(raw_path, str):
            raise ValueError(f"disk_cache_path must be a string, not {raw_path!r}")

        stripped = raw_path.strip()
        if not stripped:
            return cls(enabled=False, directory=None, budget_bytes=0, refused_type=refused_type, idle_save_s=idle_save_s)
        if _is_network_or_invalid_path(stripped):
            raise ValueError(f"disk_cache_path cannot be a network/UNC path or contain NUL: {raw_path!r}")

        if gib == 0:
            return cls(enabled=False, directory=None, budget_bytes=0, refused_type=refused_type, idle_save_s=idle_save_s)

        resolved = stripped if os.path.isabs(stripped) else os.path.abspath(os.path.join(cwd or os.getcwd(), stripped))
        target = Path(resolved)
        parent = target.parent

        # Never create parent directories across missing/unmounted mountpoints.
        if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
            print(
                f"[strata] disk session cache disabled: parent directory missing or not writable ({resolved})",
                flush=True,
            )
            return cls(enabled=False, directory=None, budget_bytes=0, refused_type=refused_type, idle_save_s=idle_save_s)

        if target.is_symlink() or target.exists():
            if target.is_symlink() or not target.is_dir() or not os.access(target, os.W_OK | os.X_OK):
                print(
                    f"[strata] disk session cache disabled: target is not a writable directory ({resolved})",
                    flush=True,
                )
                return cls(
                    enabled=False, directory=None, budget_bytes=0, refused_type=refused_type, idle_save_s=idle_save_s
                )
        else:
            try:
                os.mkdir(resolved, 0o700)
            except OSError as exc:
                print(
                    f"[strata] disk session cache disabled: could not create directory ({exc}) ({resolved})",
                    flush=True,
                )
                return cls(
                    enabled=False, directory=None, budget_bytes=0, refused_type=refused_type, idle_save_s=idle_save_s
                )

        budget_bytes = max(1, int(gib * (1 << 30)))
        return cls(
            enabled=True,
            directory=resolved,
            budget_bytes=budget_bytes,
            refused_type=refused_type,
            idle_save_s=idle_save_s,
        )

    def _index_path(self) -> Path:
        assert self.dir is not None
        return Path(self.dir) / INDEX_NAME

    def _bin_path(self, filename: str) -> str:
        assert self.dir is not None
        return os.path.join(self.dir, filename)

    def _load_and_reconcile_index(self) -> None:
        assert self.dir is not None
        idx_path = self._index_path()
        raw_entries: list[dict] = []
        dirty_index = False

        if idx_path.exists():
            try:
                if idx_path.is_symlink():
                    raise ValueError("index.json is a symlink")
                data = json.loads(idx_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
                    raise ValueError("invalid index schema")
                raw_entries = data["entries"]
            except (OSError, ValueError) as exc:
                print(f"[strata] disk cache: ignoring unreadable index.json ({exc}); starting fresh", flush=True)
                raw_entries = []
                dirty_index = True

        valid: list[dict] = []
        referenced_files: set[str] = set()
        for item in raw_entries:
            if not isinstance(item, dict):
                dirty_index = True
                continue
            fname = item.get("filename")
            tokens = item.get("tokens")
            if (
                not isinstance(fname, str)
                or not fname.startswith(SESSION_PREFIX)
                or not fname.endswith(SESSION_SUFFIX)
                or "/" in fname
                or "\\" in fname
                or not isinstance(tokens, list)
                or not tokens
            ):
                dirty_index = True
                continue
            bpath = Path(self.dir) / fname
            try:
                st = os.lstat(bpath)
            except OSError:
                dirty_index = True
                continue
            if not stat.S_ISREG(st.st_mode) or st.st_size <= 0:
                _safe_unlink(bpath)
                dirty_index = True
                continue
            if item.get("bytes") != st.st_size:
                dirty_index = True
            try:
                if any(isinstance(t, bool) for t in tokens):
                    raise ValueError("boolean token in index")
                coerced_tokens = [int(t) for t in tokens]
                raw_plen = item.get("prompt_len")
                if isinstance(raw_plen, bool):
                    raise ValueError("boolean prompt_len in index")
                prompt_len = len(coerced_tokens) if raw_plen is None else max(0, min(int(raw_plen), len(coerced_tokens)))
                last_used = float(item.get("last_used") or st.st_mtime)
                created_at = float(item.get("created_at") or st.st_mtime)
                if not (math.isfinite(last_used) and math.isfinite(created_at)):
                    raise ValueError("non-finite timestamp in index")
            except (TypeError, ValueError):
                _safe_unlink(bpath)
                dirty_index = True
                continue
            valid.append(
                {
                    "id": str(item.get("id") or fname[len(SESSION_PREFIX) : -len(SESSION_SUFFIX)]),
                    "filename": fname,
                    "tokens": coerced_tokens,
                    "prompt_len": prompt_len,
                    "bytes": int(st.st_size),
                    "last_used": last_used,
                    "created_at": created_at,
                }
            )
            referenced_files.add(fname)

        self.entries = valid

        # Remove orphan session-*.bin and leftover .tmp files.
        try:
            for child in os.listdir(self.dir):
                if child == INDEX_NAME or child in referenced_files:
                    continue
                if (child.startswith(SESSION_PREFIX) and child.endswith(SESSION_SUFFIX)) or child.endswith(".tmp"):
                    _safe_unlink(os.path.join(self.dir, child))
        except OSError:
            pass

        if self._evict_to_budget():
            dirty_index = True

        if dirty_index:
            self._write_index()

    def _write_index(self) -> None:
        if not self.enabled or self.dir is None:
            return
        payload = json.dumps({"version": 1, "entries": self.entries}, separators=(",", ":")).encode("utf-8")
        tmp_path: str | None = None
        try:
            fd, tmp_path = tempfile.mkstemp(prefix=".index-", suffix=".tmp", dir=self.dir)
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
            try:
                os.chmod(tmp_path, 0o600)
            except OSError:
                pass
            os.replace(tmp_path, self._index_path())
            tmp_path = None
        except OSError as exc:
            print(f"[strata] disk cache: could not update index.json ({exc})", flush=True)
        finally:
            if tmp_path is not None:
                _safe_unlink(tmp_path)

    def _evict_to_budget(self) -> bool:
        total = sum(e["bytes"] for e in self.entries)
        if total <= self.budget_bytes:
            return False
        # Evict least-recently-used entries first.
        by_lru = sorted(self.entries, key=lambda e: (e["last_used"], e["created_at"]))
        evicted_ids: set[str] = set()
        for entry in by_lru:
            if total <= self.budget_bytes:
                break
            _safe_unlink(self._bin_path(entry["filename"]))
            total -= entry["bytes"]
            evicted_ids.add(entry["filename"])
            print(
                f"[strata] disk cache: evicted {entry['filename']} "
                f"({entry['bytes'] / (1 << 20):.1f} MiB, {len(entry['tokens'])} tokens) to fit budget",
                flush=True,
            )
        if evicted_ids:
            self.entries = [e for e in self.entries if e["filename"] not in evicted_ids]
            self.evictions += len(evicted_ids)
            return True
        return False

    def prune_incompatible(self, engine_info: dict | None) -> int:
        """Delete indexed `.bin` files whose header/trailer or `(version, model, config)` fingerprint does not match the live engine."""
        if not self.enabled or self.dir is None or not isinstance(engine_info, dict):
            return 0
        fp = _parse_session_fp(engine_info.get("session_fp"))
        if fp is None or not self.entries:
            return 0
        want_ver, want_model, want_cfg = fp
        kept: list[dict] = []
        pruned = 0
        freed_bytes = 0
        for entry in self.entries:
            bpath = self._bin_path(entry["filename"])
            if _session_bin_compatible(bpath, want_ver, want_model, want_cfg):
                kept.append(entry)
            else:
                _safe_unlink(bpath)
                pruned += 1
                freed_bytes += int(entry.get("bytes") or 0)
        if pruned:
            self.entries = kept
            self.evictions += pruned
            self._write_index()
            self._publish_view()
            print(
                f"[strata] disk cache: pruned {pruned} incompatible session file(s) "
                f"({freed_bytes / (1 << 20):.1f} MiB)",
                flush=True,
            )
        return pruned

    def _publish_view(self) -> None:
        used_bytes = sum(int(e.get("bytes") or 0) for e in self.entries)
        total_tokens = sum(len(e.get("tokens") or ()) for e in self.entries)
        ordered = sorted(self.entries, key=lambda e: (e.get("last_used", 0.0), e.get("created_at", 0.0)), reverse=True)
        entry_summaries = [
            {
                "id": str(e.get("id") or ""),
                "tokens": len(e.get("tokens") or ()),
                "bytes": int(e.get("bytes") or 0),
                "last_used": float(e.get("last_used") or 0.0),
            }
            for e in ordered[:12]
        ]
        live_len = len(self.live_tokens)
        unsaved = max(0, live_len - self.last_saved_len) if self.live_dirty else 0
        view = {
            "enabled": bool(self.enabled),
            "dir": self.dir,
            "budget_bytes": int(self.budget_bytes),
            "used_bytes": int(used_bytes),
            "sessions": len(self.entries),
            "total_tokens": int(total_tokens),
            "live_tokens": int(live_len),
            "live_dirty": bool(self.live_dirty),
            "live_unsaved_tokens": int(unsaved),
            "periodic_save_tokens": PERIODIC_SAVE_TOKENS,
            "saves": int(self.saves),
            "restores": int(self.restores),
            "evictions": int(self.evictions),
            "restored_tokens": int(self.restored_tokens),
            "last_event": dict(self.last_event) if self.last_event is not None else None,
            "entries": entry_summaries,
        }
        with self._stats_lock:
            self._cached_view = view

    def metrics_view(self) -> dict:
        with self._stats_lock:
            return dict(self._cached_view)

    def _remember_service(self, svc) -> None:
        try:
            self._svc_ref = weakref.ref(svc)
        except TypeError:
            self._svc_ref = lambda: svc  # type: ignore[assignment]

    def _sync_engine_gen(self, svc) -> None:
        cur_gen = int(getattr(svc.engine, "gen", 0) or 0)
        if cur_gen != self.live_gen:
            self.live_tokens = []
            self.live_prompt_len = 0
            self.live_dirty = False
            self.last_saved_len = 0
            self.live_gen = cur_gen
            self._publish_view()

    def invalidate_live(self) -> None:
        self.live_tokens = []
        self.live_prompt_len = 0
        self.live_dirty = False
        self.last_saved_len = 0
        with self._cv:
            self._idle_deadline = None
        self._publish_view()

    def _recover_engine_if_dead(self, svc, exc: BaseException) -> bool:
        engine_died = (not svc.loaded()) or exc.__class__.__name__ in ("EngineDied", "EngineSilent")
        if not engine_died:
            return False
        if hasattr(svc.engine, "ended"):
            svc.engine.ended = True
        self.invalidate_live()
        if hasattr(svc, "ensure_loaded"):
            svc.ensure_loaded()
        self._sync_engine_gen(svc)
        return True

    def before_multimodal(self, svc) -> None:
        if (
            not self.enabled
            or getattr(svc.engine, "batch", 0)
            or not hasattr(svc.engine, "session_file")
        ):
            return
        self._remember_service(svc)
        with self._cv:
            self._idle_deadline = None
        self._sync_engine_gen(svc)
        if self.live_dirty and len(self.live_tokens) >= MIN_SAVE_TOKENS and svc.loaded():
            self._save_live_locked(svc, reason="switch", recover_dead=True)
        self.invalidate_live()

    def before_generate(self, svc, prompt_ids: list[int]) -> dict | None:
        if (
            not self.enabled
            or getattr(svc.engine, "batch", 0)
            or not hasattr(svc.engine, "session_file")
        ):
            return None

        self._remember_service(svc)
        with self._cv:
            self._idle_deadline = None

        self._sync_engine_gen(svc)
        live_lcp = lcp_len(self.live_tokens, prompt_ids)

        ranked: list[tuple[int, dict]] = []
        for entry in self.entries:
            match_len = lcp_len(entry["tokens"], prompt_ids)
            if match_len >= MIN_RESTORE_GAIN:
                ranked.append((match_len, entry))
        ranked.sort(key=lambda pair: (pair[0], -len(pair[1]["tokens"]), pair[1]["last_used"]), reverse=True)

        if ranked and (ranked[0][0] - live_lcp) >= MIN_RESTORE_GAIN:
            if self.live_dirty and len(self.live_tokens) >= MIN_SAVE_TOKENS:
                self._save_live_locked(svc, reason="switch", recover_dead=True)
                live_lcp = lcp_len(self.live_tokens, prompt_ids)

            stale_files: set[str] = set()
            restored_summary: dict | None = None
            for match_len, entry in ranked:
                if (match_len - live_lcp) < MIN_RESTORE_GAIN:
                    break
                bpath = self._bin_path(entry["filename"])
                try:
                    if os.path.islink(bpath) or not os.path.isfile(bpath):
                        raise FileNotFoundError(entry["filename"])
                    res = svc.engine.session_file("restore", bpath)
                except self.refused_type as exc:
                    kind = getattr(exc, "kind", "invalid")
                    if kind == "memory":
                        print(f"[strata] disk cache: restore skipped due to RAM headroom ({exc})", flush=True)
                        break
                    print(f"[strata] disk cache: evicting unreadable/stale session {entry['filename']} ({exc})", flush=True)
                    _safe_unlink(bpath)
                    stale_files.add(entry["filename"])
                    continue
                except OSError as exc:
                    print(f"[strata] disk cache: evicting missing session {entry['filename']} ({exc})", flush=True)
                    _safe_unlink(bpath)
                    stale_files.add(entry["filename"])
                    continue
                except Exception as exc:  # noqa: BLE001 - never fail chat request on cache error
                    if self._recover_engine_if_dead(svc, exc):
                        print(
                            f"[strata] disk cache: evicting corrupt session {entry['filename']} after engine exit ({exc})",
                            flush=True,
                        )
                        _safe_unlink(bpath)
                        stale_files.add(entry["filename"])
                    else:
                        print(f"[strata] disk cache: restore failed ({exc})", flush=True)
                    break

                now_ts = time.time()
                entry["last_used"] = now_ts
                self.live_tokens = list(entry["tokens"])
                self.live_prompt_len = int(entry.get("prompt_len") or len(self.live_tokens))
                self.live_dirty = False
                self.last_saved_len = match_len
                self.live_gen = int(getattr(svc.engine, "gen", 0) or 0)
                n_tok = int(res.get("tokens", len(self.live_tokens)))
                n_bytes = int(res.get("bytes", entry["bytes"]))
                ms = float(res.get("ms", 0.0))
                self.restores += 1
                self.restored_tokens += match_len
                self.last_event = {
                    "kind": "restored",
                    "tokens": n_tok,
                    "matched_tokens": match_len,
                    "bytes": n_bytes,
                    "ms": round(ms, 1),
                    "at": now_ts,
                }
                print(
                    f"[strata] disk cache: restored {n_tok} tokens ({n_bytes / (1 << 20):.1f} MiB in {ms:.0f} ms, "
                    f"prefix match {match_len} tokens)",
                    flush=True,
                )
                restored_summary = {"filename": entry["filename"], "tokens": n_tok, "bytes": n_bytes, "lcp": match_len}
                break

            if stale_files:
                self.entries = [e for e in self.entries if e["filename"] not in stale_files]
                self.evictions += len(stale_files)
            if stale_files or restored_summary is not None:
                self._write_index()
                self._publish_view()
            return restored_summary

        # Switching away from a dirty long conversation when no disk session was restored.
        # On reasoning models, clients strip `<think>...</think>` on Turn N+1 so `live_lcp` stops near
        # `live_prompt_len` even though `live_tokens` also contains Turn N's `<think>` + answer tokens.
        continuing_live = (
            self.live_prompt_len > 0
            and live_lcp >= max(0, self.live_prompt_len - THINK_STRIP_SLACK)
            and len(prompt_ids) > self.live_prompt_len
        )
        if (
            self.live_dirty
            and not continuing_live
            and len(self.live_tokens) >= MIN_SAVE_TOKENS
            and (live_lcp + MIN_RESTORE_GAIN) < len(self.live_tokens)
        ):
            self._save_live_locked(svc, reason="switch", recover_dead=True)

        return None

    def after_generate(
        self,
        svc,
        full_tokens: list[int],
        *,
        ok: bool,
        prompt_len: int | None = None,
    ) -> None:
        if (
            not self.enabled
            or getattr(svc.engine, "batch", 0)
            or not hasattr(svc.engine, "session_file")
        ):
            return

        self._remember_service(svc)
        if not ok:
            self.invalidate_live()
            return

        self.live_gen = int(getattr(svc.engine, "gen", 0) or 0)
        saved_base = min(self.last_saved_len, lcp_len(full_tokens, self.live_tokens)) if self.last_saved_len > 0 else 0

        self.live_tokens = list(full_tokens)
        self.live_prompt_len = (
            len(self.live_tokens) if prompt_len is None else max(0, min(int(prompt_len), len(self.live_tokens)))
        )
        self.last_saved_len = saved_base
        unsaved = len(self.live_tokens) - self.last_saved_len
        self.live_dirty = len(self.live_tokens) >= MIN_SAVE_TOKENS and unsaved >= MIN_SAVE_GAIN

        if self.live_dirty and unsaved >= PERIODIC_SAVE_TOKENS:
            self._save_live_locked(svc, reason="checkpoint")
        elif self.live_dirty:
            self._arm_idle()
        self._publish_view()

    def on_unload_locked(self, svc) -> None:
        if not self.enabled:
            return
        if self.live_dirty and len(self.live_tokens) >= MIN_SAVE_TOKENS and svc.loaded():
            self._save_live_locked(svc, reason="unload")
        self.invalidate_live()

    def _call_engine_save(self, svc, path: str) -> dict:
        if getattr(svc.engine, "session_save_reclaim", False):
            from serve.session_save_retry import session_file as retry_session_file

            return retry_session_file(svc, "save", path, self.refused_type)
        return svc.engine.session_file("save", path)

    def _save_live_locked(self, svc, *, reason: str, recover_dead: bool = False) -> bool:
        if (
            not self.enabled
            or self.dir is None
            or not self.live_dirty
            or len(self.live_tokens) < MIN_SAVE_TOKENS
            or not svc.loaded()
        ):
            return False

        sid = secrets.token_hex(6)
        fname = f"{SESSION_PREFIX}{sid}{SESSION_SUFFIX}"
        bpath = self._bin_path(fname)

        try:
            try:
                res = self._call_engine_save(svc, bpath)
            except self.refused_type as exc:
                if getattr(exc, "kind", None) == "storage" and self.entries:
                    _safe_unlink(bpath)
                    oldest = min(self.entries, key=lambda e: (e["last_used"], e["created_at"]))
                    _safe_unlink(self._bin_path(oldest["filename"]))
                    self.entries = [e for e in self.entries if e["filename"] != oldest["filename"]]
                    self.evictions += 1
                    self._write_index()
                    self._publish_view()
                    res = self._call_engine_save(svc, bpath)
                else:
                    raise
        except Exception as exc:  # noqa: BLE001 - never fail chat request on cache error
            _safe_unlink(bpath)
            print(f"[strata] disk cache: save skipped ({exc})", flush=True)
            if recover_dead:
                self._recover_engine_if_dead(svc, exc)
            return False

        try:
            actual_bytes = os.lstat(bpath).st_size
        except OSError as exc:
            print(f"[strata] disk cache: saved file disappeared ({exc})", flush=True)
            return False

        if actual_bytes > self.budget_bytes:
            _safe_unlink(bpath)
            self.live_dirty = False
            self.last_saved_len = len(self.live_tokens)
            with self._cv:
                self._idle_deadline = None
            print(
                f"[strata] disk cache: session ({actual_bytes / (1 << 20):.1f} MiB) exceeds disk_cache_gib budget "
                f"({self.budget_bytes / (1 << 30):.2f} GiB); not kept",
                flush=True,
            )
            return False

        # Supersede older snapshots that are prefixes (including stripped-<think> continuations) of this conversation.
        cur_prompt_len = self.live_prompt_len or len(self.live_tokens)
        superseded: set[str] = set()
        for existing in self.entries:
            common = lcp_len(existing["tokens"], self.live_tokens)
            existing_len = len(existing["tokens"])
            existing_prompt_len = int(existing.get("prompt_len") or existing_len)
            if (
                common == existing_len
                or (
                    common >= MIN_SAVE_TOKENS
                    and (existing_len - common) < MIN_SAVE_GAIN
                    and len(self.live_tokens) >= existing_len
                )
                or (
                    common >= MIN_SAVE_TOKENS
                    and common >= max(0, existing_prompt_len - THINK_STRIP_SLACK)
                    and cur_prompt_len > existing_prompt_len
                )
            ):
                _safe_unlink(self._bin_path(existing["filename"]))
                superseded.add(existing["filename"])
        if superseded:
            self.entries = [e for e in self.entries if e["filename"] not in superseded]

        now = time.time()
        self.entries.append(
            {
                "id": sid,
                "filename": fname,
                "tokens": list(self.live_tokens),
                "prompt_len": int(cur_prompt_len),
                "bytes": int(actual_bytes),
                "last_used": now,
                "created_at": now,
            }
        )
        self._evict_to_budget()
        self._write_index()

        self.live_dirty = False
        self.last_saved_len = len(self.live_tokens)
        with self._cv:
            self._idle_deadline = None

        ms = float(res.get("ms", 0.0))
        self.saves += 1
        self.last_event = {
            "kind": "saved",
            "tokens": len(self.live_tokens),
            "bytes": int(actual_bytes),
            "ms": round(ms, 1),
            "reason": reason,
            "at": now,
        }
        self._publish_view()
        print(
            f"[strata] disk cache: saved {len(self.live_tokens)} tokens "
            f"({actual_bytes / (1 << 20):.1f} MiB in {ms:.0f} ms, reason={reason})",
            flush=True,
        )
        return True

    def _arm_idle(self) -> None:
        with self._cv:
            if self._closing:
                return
            self._idle_deadline = time.monotonic() + self.idle_save_s
            if self._idle_thread is None:
                self._idle_thread = threading.Thread(target=self._idle_loop, daemon=True, name="strata-disk-cache-idle")
                self._idle_thread.start()
            else:
                self._cv.notify_all()

    def _idle_loop(self) -> None:
        while True:
            with self._cv:
                while not self._closing and self._idle_deadline is None:
                    self._cv.wait()
                if self._closing:
                    return
                assert self._idle_deadline is not None
                remaining = self._idle_deadline - time.monotonic()
                if remaining > 0:
                    self._cv.wait(timeout=remaining)
                    continue
                self._idle_deadline = None
                svc = self._svc_ref() if self._svc_ref is not None else None

            if svc is None:
                continue
            if not svc.fifo.acquire(blocking=False):
                self._arm_idle()
                continue
            try:
                with svc.status_lock:
                    busy = bool(svc.status.get("busy") or svc.status.get("queued"))
                if busy:
                    self._arm_idle()
                elif self.live_dirty and svc.loaded():
                    self._save_live_locked(svc, reason="idle")
            finally:
                svc.fifo.release()

    def close(self, svc=None) -> None:
        with self._cv:
            self._closing = True
            self._idle_deadline = None
            self._cv.notify_all()
            worker = self._idle_thread

        if worker is not None and worker.is_alive() and threading.current_thread() is not worker:
            worker.join(timeout=2.0)

        target_svc = svc if svc is not None else (self._svc_ref() if self._svc_ref is not None else None)
        if not self.enabled or target_svc is None:
            return
        if target_svc.fifo.acquire(timeout=CLOSE_WAIT_S):
            try:
                if self.live_dirty and target_svc.loaded():
                    self._save_live_locked(target_svc, reason="shutdown")
            finally:
                target_svc.fifo.release()
