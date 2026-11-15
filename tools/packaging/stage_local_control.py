#!/usr/bin/env python3
"""Stage and activate the local checkout as a Strata Control component (`components/control/<version>`).

Reuses the verified standalone CPython + hash-locked wheelset from an already-installed `components/control/*`
bundle whenever `PYTHON_PINS[target]["sha256"]` and `sha256(packaging/locks/runtime-<target>-cp312.txt)` match
`packaging/package_terminal.py` (fast zero-network local copy in ~1 s). If upstream updates either pin, falls back
automatically to fetching the pinned CPython and installing the hash-locked wheels via `package_terminal.py`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import package_terminal as pt  # noqa: E402
from strata_control import atomic as _atomic_mod  # noqa: E402
from strata_control import folder_use  # noqa: E402

_HANDOVER_VER = re.compile(r"^[0-9]+(\.[0-9]+){1,3}(-[0-9A-Za-z][0-9A-Za-z.]*)?$")


def compute_local_version(base_ver: str) -> str:
    """Derive a 4-part local version (`0.1.60` -> `0.1.60.100`, `0.1.60.1` -> `0.1.60.101`).

    Must satisfy `strata_cli/handover.py` (2 to 4 numeric dot-parts) while sorting strictly above `base_ver`
    and strictly below the next upstream patch/minor version.
    """
    parts = [int(p) for p in base_ver.strip().split(".")]
    if len(parts) == 2:
        out = f"{parts[0]}.{parts[1]}.0.100"
    elif len(parts) == 3:
        out = f"{parts[0]}.{parts[1]}.{parts[2]}.100"
    elif len(parts) == 4:
        out = f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3] + 100}"
    else:
        raise SystemExit(f"unsupported base version format in CMakeLists.txt: {base_ver!r}")
    if not _HANDOVER_VER.match(out):
        raise SystemExit(f"computed version {out!r} rejected by handover version pattern")
    return out


def default_components_dir() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg) if xdg else (Path.home() / ".local" / "share")
    return base / "strata" / "components"


def _find_reusable_runtime(
    control_root: Path,
    target: str,
    want_py_sha: str,
    want_lock_sha: str,
    py_rel: str,
) -> tuple[Path, dict[str, str]] | None:
    if not control_root.is_dir():
        return None
    candidates = sorted(
        (d for d in control_root.iterdir() if d.is_dir() and not d.name.startswith(".")),
        key=lambda d: [int(x) if x.isdigit() else x for x in re.split(r"([0-9]+)", d.name)],
        reverse=True,
    )
    for cand in candidates:
        bfile = cand / "BUNDLE.json"
        py_bin = cand / py_rel
        if not (bfile.is_file() and py_bin.is_file()):
            continue
        try:
            data = json.loads(bfile.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        py_info = data.get("python") if isinstance(data.get("python"), dict) else {}
        lock_info = data.get("lock") if isinstance(data.get("lock"), dict) else {}
        if (
            data.get("target") == target
            and py_info.get("sha256") == want_py_sha
            and lock_info.get("sha256") == want_lock_sha
        ):
            installed = data.get("installed") if isinstance(data.get("installed"), dict) else {}
            return cand, {str(k): str(v) for k, v in installed.items()}
    return None


def _copy_payload(stage: Path, target: str, version: str, commit: str | None) -> dict:
    for item in pt.PAYLOAD:
        src = ROOT / item
        dst = stage / item
        if not src.exists():
            raise SystemExit(f"missing payload item: {item}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=pt.ignore)
        else:
            shutil.copy2(src, dst)

    tools_dir = stage / "tools"
    tools_dir.mkdir(exist_ok=True)
    for item in pt.TOOLS_SUBSET:
        src = ROOT / "tools" / item
        if not src.exists():
            raise SystemExit(f"missing tools module: {item}")
        shutil.copy2(src, tools_dir / item)

    installer = pt.add_installer(stage, ROOT)
    (stage / "VERSION").write_text(version + "\n", encoding="utf-8")
    if commit:
        (stage / "COMMIT").write_text(commit + "\n", encoding="utf-8")
    (stage / "README.txt").write_text(pt._readme(version, target), encoding="utf-8")

    bin_dir = stage / "bin"
    bin_dir.mkdir(exist_ok=True)
    launchers = ["strata.cmd", "strata.ps1"] if target.startswith("windows") else ["strata"]
    for launcher in launchers:
        shutil.copy2(ROOT / "bin" / launcher, bin_dir / launcher)
        if launcher == "strata":
            (bin_dir / launcher).chmod(0o755)

    return installer


def _verify_staged(py: Path, stage: Path, version: str) -> None:
    probe = pt.run_probe(py, stage)
    if probe.returncode != 0:
        raise SystemExit(
            f"bundled interpreter failed runtime import probe:\n{probe.stdout}\n{probe.stderr}"
        )

    clean_env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "STRATA_NO_ACQUIRE": "1",
        "STRATA_HANDOVER": "1",
    }
    for k in ("PYTHONHOME", "PYTHONPATH"):
        clean_env.pop(k, None)

    ver_run = subprocess.run(
        [str(py), "-B", str(stage / "strata.py"), "--version"],
        capture_output=True,
        text=True,
        cwd=str(stage),
        env=clean_env,
        timeout=60,
    )
    if ver_run.returncode != 0 or version not in ver_run.stdout:
        raise SystemExit(
            f"staged Control --version check failed (expected {version!r}):\n"
            f"stdout={ver_run.stdout!r}\nstderr={ver_run.stderr!r}"
        )

    inst_code = (
        "import sys, importlib.util as u; "
        f"sys.path.insert(0, {str(stage / 'tools')!r}); "
        f"s = u.spec_from_file_location('s', {str(stage / 'setup.py')!r}); "
        "m = u.module_from_spec(s); s.loader.exec_module(m); "
        "import iq_pack, strata_pack, pack_index, mtp_fetch; "
        "print('installer ok')"
    )
    inst_env = {**clean_env, "STRATA_CONTROL_INSTALL": "1"}
    inst_run = subprocess.run(
        [str(py), "-B", "-c", inst_code],
        capture_output=True,
        text=True,
        cwd=str(stage),
        env=inst_env,
        timeout=120,
    )
    if inst_run.returncode != 0:
        raise SystemExit(
            f"staged Control installer probe failed:\n{inst_run.stdout}\n{inst_run.stderr[-800:]}"
        )


def _update_state_record(components: Path, previous_active: dict | None, version: str) -> None:
    state_path = components / ".updates" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if state_path.is_file():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            data = {}
    data["schema_version"] = 1
    ctrl = data.get("control")
    if not isinstance(ctrl, dict):
        ctrl = {}
        data["control"] = ctrl
    if previous_active and previous_active.get("folder") != version:
        ctrl["previous"] = previous_active
    history = data.setdefault("history", [])
    if isinstance(history, list):
        history.append(
            {
                "at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "action": "stage-local-control",
                "from": (previous_active or {}).get("version"),
                "to": version,
                "ok": True,
            }
        )
        del history[:-30]
    fd, tmp = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=state_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
            fh.write("\n")
        _atomic_mod.replace(tmp, state_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--components", type=Path, default=None, help="Strata components directory")
    ap.add_argument("--version", default=None, help="4-part Control version to stage (default: derived from CMakeLists.txt)")
    ap.add_argument("--commit", default=None, help="40-hex source commit (default: git rev-parse HEAD)")
    ap.add_argument("--cache", type=Path, default=None, help="cache folder if CPython archive download is needed")
    ap.add_argument("--wheelhouse", type=Path, default=None, help="optional offline wheelhouse if lock changed")
    args = ap.parse_args()

    components = (args.components or default_components_dir()).resolve()
    control_root = components / "control"
    control_root.mkdir(parents=True, exist_ok=True)

    version = args.version or compute_local_version(pt._source_version())
    if not _HANDOVER_VER.match(version):
        raise SystemExit(
            f"version {version!r} does not match strata_cli/handover.py 2..4 part requirement"
        )

    target = pt.host_target()
    pin = pt.PYTHON_PINS[target]
    lock = pt.lock_for(target)
    want_py_sha = pin["sha256"]
    want_lock_sha = pt.sha256(lock)
    commit = pt.build_commit(args.commit)

    reusable = _find_reusable_runtime(control_root, target, want_py_sha, want_lock_sha, pin["python"])
    staging_parent = Path(tempfile.mkdtemp(prefix=".stage-control-", dir=control_root))
    stage = staging_parent / version
    stage.mkdir(parents=True)
    try:
        if reusable is not None:
            donor_dir, installed = reusable
            print(f"==> Reusing verified Python {pt.PYTHON_VERSION} runtime from {donor_dir}", flush=True)
            shutil.copytree(donor_dir / "python", stage / "python", symlinks=True, ignore=pt.ignore)
        else:
            cache_dir = (args.cache or (components.parent / ".packaging-cache")).resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"==> Pin change detected; building fresh Python {pt.PYTHON_VERSION} runtime", flush=True)
            archive = pt.fetch_python(target, cache_dir)
            with tarfile.open(archive) as tf:
                tf.extractall(stage)
            py_fresh = stage / pin["python"]
            installed = pt.install_locked(
                py_fresh, lock, stage, args.wheelhouse.resolve() if args.wheelhouse else None
            )
            pt.drop_build_path_launchers(stage, staging_parent, target)

        installer = _copy_payload(stage, target, version, commit)
        for pycache in list(stage.rglob("__pycache__")):
            shutil.rmtree(pycache, ignore_errors=True)

        bundle_record = {
            "schema_version": 1,
            "component": "strata-control",
            "version": version,
            "target": target,
            "python": {
                "version": pt.PYTHON_VERSION,
                "build": pt.PBS_RELEASE,
                "sha256": want_py_sha,
            },
            "lock": {
                "file": lock.relative_to(ROOT).as_posix(),
                "sha256": want_lock_sha,
                "install": "pip install --require-hashes --no-deps --only-binary=:all: (build time only)",
            },
            "installed": installed,
            "source": "packaging/stage_local_control.py",
            "commit": commit,
            "acquires_at_runtime": False,
            "installer": installer,
        }
        bundle_bytes = (json.dumps(bundle_record, indent=1) + "\n").encode("utf-8")
        (stage / "BUNDLE.json").write_bytes(bundle_bytes)
        bundle_sha = hashlib.sha256(bundle_bytes).hexdigest()

        py = stage / pin["python"]
        _verify_staged(py, stage, version)

        previous_active: dict | None = None
        active_path = control_root / "active.json"
        if active_path.is_file():
            try:
                parsed = json.loads(active_path.read_text(encoding="utf-8-sig"))
                if isinstance(parsed, dict) and parsed.get("version") and parsed.get("folder"):
                    previous_active = parsed
            except (OSError, ValueError):
                previous_active = None

        dest = control_root / version
        old_dest: Path | None = None
        if dest.exists():
            if folder_use.in_use(dest):
                # Never delete files underneath a live running server; copy alongside or swap directory entry only.
                old_dest = control_root / f".{version}.inuse.{os.getpid()}"
            else:
                old_dest = control_root / f".{version}.old.{os.getpid()}"
            dest.rename(old_dest)
        stage.rename(dest)
        if old_dest is not None and not folder_use.in_use(old_dest):
            shutil.rmtree(old_dest, ignore_errors=True)

        _update_state_record(components, previous_active, version)

        tmp_active = control_root / f".active.{os.getpid()}.tmp"
        tmp_active.write_text(
            json.dumps({"version": version, "folder": version, "sha256": bundle_sha}),
            encoding="utf-8",
        )
        _atomic_mod.replace(tmp_active, active_path)

        # Clean up superseded local `.10X` control folders that are not active, previous, or in use.
        keep = {version}
        if previous_active and previous_active.get("folder"):
            keep.add(str(previous_active["folder"]))
        for entry_dir in control_root.iterdir():
            if (
                entry_dir.is_dir()
                and not entry_dir.name.startswith(".")
                and entry_dir.name not in keep
                and re.match(r"^[0-9]+\.[0-9]+\.[0-9]+\.1[0-9]{2,}$", entry_dir.name)
                and not folder_use.in_use(entry_dir)
            ):
                shutil.rmtree(entry_dir, ignore_errors=True)

        print(f"==> Activated Control component: {dest} (active.json -> {version})", flush=True)
        return 0
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
