#!/usr/bin/env bash
# Build and install BOTH the local CUDA 12 Strata engine component (default sm_61 / Pascal)
# and the matching Strata Control component directly from this git checkout.
#
# Typical usage after an upstream Strata release:
#   ~/.local/share/strata/rebuild-strata-engine.sh --rebase --push --run-tests
#
# Quick incremental rebuild of current checkout:
#   ~/.local/share/strata/rebuild-strata-engine.sh --run-tests

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$(readlink -f -- "$0")")" && pwd -P)"
STRATA_REPO="${STRATA_REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}"
XDG_DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
STRATA_DATA_DIR="${STRATA_DATA_DIR:-$XDG_DATA/strata}"
COMPONENTS_DIR="${STRATA_COMPONENTS_DIR:-$STRATA_DATA_DIR/components}"
BUILDER_IMAGE="${STRATA_BUILDER_IMAGE:-strata-cuda12-builder:12.9.1}"
CUDA_ARCH="${STRATA_CUDA_ARCH:-61}"
UPSTREAM_URL="${STRATA_UPSTREAM_URL:-https://github.com/Niko1221/Strata.git}"

DO_REBASE=0
DO_PUSH=0
RUN_TESTS=0
CLEAN_BUILD=0
REBUILD_IMAGE=0
BUILD_VISION=0

usage() {
    cat <<'EOF'
Usage: rebuild_local.sh [options]

Options:
  --rebase         Fetch upstream (Niko1221/Strata) and rebase current branch onto upstream/main
  --push           Push current branch to origin (--force-with-lease) after build & tests succeed
  --run-tests      Build and run C++ session/snapshot tests and full Python bundle/control tests
  --build-vision   Compile strata-vision from source instead of reusing installed strata-vision
  --clean          Remove build-cuda12-sm<ARCH> before configuring (keeps .ccache)
  --rebuild-image  Rebuild the cached Docker CUDA 12 builder image
  --arch <N>       Target CUDA compute capability (default: 61 for GTX 1070 / Pascal)
  -h, --help       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rebase)        DO_REBASE=1; shift ;;
        --push)          DO_PUSH=1; shift ;;
        --run-tests)     RUN_TESTS=1; shift ;;
        --build-vision)  BUILD_VISION=1; shift ;;
        --clean)         CLEAN_BUILD=1; shift ;;
        --rebuild-image) REBUILD_IMAGE=1; shift ;;
        --arch)          CUDA_ARCH="${2:?--arch requires a value}"; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *)               echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ ! -d "$STRATA_REPO/.git" ]]; then
    echo "error: Strata repository not found at $STRATA_REPO" >&2
    exit 1
fi

if ! git -C "$STRATA_REPO" remote get-url upstream >/dev/null 2>&1; then
    echo "==> Adding upstream remote ($UPSTREAM_URL)"
    git -C "$STRATA_REPO" remote add upstream "$UPSTREAM_URL"
fi

if [[ "$DO_REBASE" -eq 1 ]]; then
    if [[ -n "$(git -C "$STRATA_REPO" status --porcelain --untracked-files=no)" ]]; then
        echo "error: $STRATA_REPO has uncommitted changes; commit or stash before --rebase" >&2
        exit 1
    fi
    echo "==> Fetching upstream and rebasing onto upstream/main"
    git -C "$STRATA_REPO" fetch upstream
    git -C "$STRATA_REPO" rebase upstream/main
fi

if [[ "$REBUILD_IMAGE" -eq 1 ]] || ! docker image inspect "$BUILDER_IMAGE" >/dev/null 2>&1; then
    echo "==> Building Docker toolchain image $BUILDER_IMAGE"
    docker build -t "$BUILDER_IMAGE" - <<'EOF'
FROM nvidia/cuda:12.9.1-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        git \
        libatomic1 \
        libgomp1 \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir cmake==4.4.3 ninja==1.13.2
ENV PATH="/opt/venv/bin:/usr/local/cuda/bin:${PATH}"
EOF
fi

BUILD_REL="build-cuda12-sm${CUDA_ARCH}"
BUILD_ABS="$STRATA_REPO/$BUILD_REL"
if [[ "$CLEAN_BUILD" -eq 1 && -d "$BUILD_ABS" ]]; then
    echo "==> Cleaning $BUILD_ABS"
    rm -rf "$BUILD_ABS"
fi
mkdir -p "$STRATA_REPO/.ccache"

BASE_VER="$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' "$STRATA_REPO/CMakeLists.txt" | head -n 1)"
if [[ -z "$BASE_VER" ]]; then
    echo "error: could not parse project VERSION from $STRATA_REPO/CMakeLists.txt" >&2
    exit 1
fi

# Compute a valid 4-part version (0.1.60 -> 0.1.60.100, 0.1.60.1 -> 0.1.60.101) accepted by
# both strata_cli/handover.py and strata_control/engine_component.py, sorting above BASE_VER.
PKG_VER="$(python3 - "$BASE_VER" <<'PY'
import sys
parts = [int(p) for p in sys.argv[1].strip().split(".")]
if len(parts) == 2:
    print(f"{parts[0]}.{parts[1]}.0.100")
elif len(parts) == 3:
    print(f"{parts[0]}.{parts[1]}.{parts[2]}.100")
elif len(parts) == 4:
    print(f"{parts[0]}.{parts[1]}.{parts[2]}.{parts[3] + 100}")
else:
    raise SystemExit(f"unsupported CMake VERSION {sys.argv[1]!r}")
PY
)"

GIT_FULL="$(git -C "$STRATA_REPO" rev-parse HEAD)"
GIT_HEAD="${GIT_FULL:0:12}"
if [[ -n "$(git -C "$STRATA_REPO" status --porcelain --untracked-files=no)" ]]; then
    GIT_HEAD="${GIT_HEAD}-dirty"
fi
BUILD_ID="local-${GIT_HEAD}+cuda12.9-sm${CUDA_ARCH}"

# Locate an existing compatible strata-vision binary to reuse unless --build-vision was requested
EXISTING_VISION=""
EXISTING_VISION_ARCHS="60;61;70;75;80;86;89"
if [[ "$BUILD_VISION" -eq 0 && -d "$COMPONENTS_DIR/engine" ]]; then
    while IFS= read -r cand; do
        [[ -f "$cand" ]] || continue
        cand_dir="$(dirname -- "$cand")"
        if [[ "$cand_dir" == *"${PKG_VER}-linux-x86_64-cuda-cuda12" ]]; then
            continue
        fi
        EXISTING_VISION="$cand"
        if [[ -f "$cand_dir/BUILD.json" ]]; then
            parsed_archs="$(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8-sig"))
    v = d.get("vision_archs")
    if isinstance(v, list):
        print(";".join(str(x) for x in v))
    elif isinstance(v, str) and v.strip():
        print(v.strip())
except Exception:
    pass
' "$cand_dir/BUILD.json")"
            [[ -n "$parsed_archs" ]] && EXISTING_VISION_ARCHS="$parsed_archs"
        fi
    done < <(find "$COMPONENTS_DIR/engine" -maxdepth 2 -mindepth 2 -name strata-vision 2>/dev/null | sort -V)
fi

VISION_MOUNT_ARGS=()
CONTAINER_VISION_SRC=""
if [[ -n "$EXISTING_VISION" ]]; then
    echo "==> Reusing strata-vision from $EXISTING_VISION (vision_archs=$EXISTING_VISION_ARCHS)"
    VISION_MOUNT_ARGS=(-v "$EXISTING_VISION:/ro-vision/strata-vision:ro")
    CONTAINER_VISION_SRC="/ro-vision/strata-vision"
fi

echo "==> Compiling Strata $BASE_VER -> engine & control $PKG_VER ($BUILD_ID) for sm_${CUDA_ARCH} in Docker"
docker run --rm \
    -u "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e CCACHE_DIR=/src/.ccache \
    -e BUILD_REL="$BUILD_REL" \
    -e CUDA_ARCH="$CUDA_ARCH" \
    -e PKG_VER="$PKG_VER" \
    -e BUILD_ID="$BUILD_ID" \
    -e RUN_TESTS="$RUN_TESTS" \
    -e BUILD_VISION="$BUILD_VISION" \
    -e CONTAINER_VISION_SRC="$CONTAINER_VISION_SRC" \
    -e EXISTING_VISION_ARCHS="$EXISTING_VISION_ARCHS" \
    -v "$STRATA_REPO:/src" \
    "${VISION_MOUNT_ARGS[@]}" \
    -w /src \
    "$BUILDER_IMAGE" \
    bash -euo pipefail -c '
        CUDA_FULL="$(nvcc --version | sed -n "s/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p" | head -n 1)"
        CMAKE_CONV_TESTS="OFF"
        if [[ "$RUN_TESTS" -eq 1 ]]; then
            CMAKE_CONV_TESTS="ON"
        fi

        cmake -G Ninja -S /src -B "/src/$BUILD_REL" \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
            -DSTRATA_ENABLE_CUDA=ON \
            -DSTRATA_BUILD_TESTS=OFF \
            -DSTRATA_BUILD_CONVERSATION_TESTS="$CMAKE_CONV_TESTS" \
            -DSTRATA_EXPERIMENTAL_SM60=ON \
            -DSTRATA_NATIVE_EXPERTS=ON \
            -DSTRATA_PORTABLE=ON \
            -DSTRATA_Q6K_EXPERTS=ON \
            -DSTRATA_MMQ_KQUANTS=ON \
            -DSTRATA_MMQ=ON \
            "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}-real;${CUDA_ARCH}-virtual"

        . /src/packaging/engine_targets.sh
        TARGETS="$(engine_targets cmake "/src/$BUILD_REL")"
        EXTRA_TARGETS=()
        if [[ "$RUN_TESTS" -eq 1 ]]; then
            EXTRA_TARGETS=(conversation_file_test session_save_reclaim_test conversation_snapshot_test)
        fi
        cmake --build "/src/$BUILD_REL" --target $TARGETS "${EXTRA_TARGETS[@]}" -j "$(nproc)"

        if [[ "$RUN_TESTS" -eq 1 ]]; then
            echo "==> Running CPU session-file unit tests inside container"
            "/src/$BUILD_REL/conversation_file_test"
            "/src/$BUILD_REL/session_save_reclaim_test"
        fi

        VISION_BIN=""
        VISION_ARCHS=""
        if [[ "$BUILD_VISION" -eq 1 ]]; then
            VISION_BUILD="/src/${BUILD_REL}-vision"
            cmake -G Ninja -S /src/tools/vision -B "$VISION_BUILD" \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
                -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
                -DSTRATA_VISION_CUDA=ON \
                -DSTRATA_PORTABLE=ON \
                -DGGML_CUDA_NCCL=OFF \
                "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}-real;${CUDA_ARCH}-virtual"
            cmake --build "$VISION_BUILD" --target strata-vision -j "$(nproc)"
            VISION_BIN="$VISION_BUILD/bin/strata-vision"
            VISION_ARCHS="$CUDA_ARCH"
        elif [[ -n "$CONTAINER_VISION_SRC" && -f "$CONTAINER_VISION_SRC" ]]; then
            VISION_BIN="$CONTAINER_VISION_SRC"
            VISION_ARCHS="$EXISTING_VISION_ARCHS"
        fi

        STAGE_DIR="/src/$BUILD_REL/stage-pkg"
        OUT_DIR="/src/$BUILD_REL/out-pkg"
        rm -rf "$STAGE_DIR" "$OUT_DIR"
        mkdir -p "$STAGE_DIR" "$OUT_DIR"

        cp "/src/$BUILD_REL/strata" "$STAGE_DIR/"
        if [[ -f "/src/$BUILD_REL/libstrata_kq.so" ]]; then
            cp "/src/$BUILD_REL/libstrata_kq.so" "$STAGE_DIR/"
        fi
        VISION_JSON=""
        if [[ -n "$VISION_BIN" ]]; then
            cp "$VISION_BIN" "$STAGE_DIR/strata-vision"
            chmod 0755 "$STAGE_DIR/strata-vision"
            VISION_JSON=", \"vision_archs\": \"${VISION_ARCHS}\""
        fi

        printf "{\"version\": \"%s\", \"source\": \"packaging/rebuild_local.sh\", \"archs\": [%s], \"ptx\": true, \"cuda\": \"%s\", \"portable\": true%s}\n" \
            "$PKG_VER" "$CUDA_ARCH" "$CUDA_FULL" "$VISION_JSON" > "$STAGE_DIR/BUILD.json"

        python3 /src/packaging/collect_libs.py \
            --binary "/src/$BUILD_REL/strata" \
            --search /usr/local/cuda/lib64 \
            --out "$STAGE_DIR" \
            --also "libcublasLt.so*"

        python3 /src/packaging/package_engine.py \
            --src "$STAGE_DIR" \
            --backend cuda \
            --version "$PKG_VER" \
            --out "$OUT_DIR" \
            --arch "sm_${CUDA_ARCH}" \
            --arch "compute_${CUDA_ARCH}" \
            --min-driver nvidia=575.51 \
            --variant cuda12 \
            --platform linux-x86_64 \
            --build-id "$BUILD_ID" \
            --include SYSTEM-LIBS.txt \
            --include "libstrata_kq*"

        PKG_FOLDER="$OUT_DIR/${PKG_VER}-linux-x86_64-cuda-cuda12"
        python3 /src/packaging/package_engine.py --verify "$PKG_FOLDER"
        python3 /src/packaging/binary_deps.py "$PKG_FOLDER"
        rm -rf "$STAGE_DIR"
    '

BUILT_PKG="$BUILD_ABS/out-pkg/${PKG_VER}-linux-x86_64-cuda-cuda12"
DEST_ENGINE_DIR="$COMPONENTS_DIR/engine"
DEST_PKG="$DEST_ENGINE_DIR/${PKG_VER}-linux-x86_64-cuda-cuda12"
mkdir -p "$DEST_ENGINE_DIR"

if [[ "$RUN_TESTS" -eq 1 ]]; then
    echo "==> Running GPU conversation_snapshot_test on host GPU"
    LD_LIBRARY_PATH="$BUILT_PKG${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$BUILD_ABS/conversation_snapshot_test"
fi

echo "==> Installing engine component to $DEST_PKG"
TMP_DEST="$DEST_ENGINE_DIR/.${PKG_VER}-linux-x86_64-cuda-cuda12.installing.$$"
rm -rf "$TMP_DEST"
cp -a "$BUILT_PKG" "$TMP_DEST"
python3 "$STRATA_REPO/packaging/package_engine.py" --verify "$TMP_DEST"
if [[ -d "$DEST_PKG" ]]; then
    OLD_DEST="$DEST_ENGINE_DIR/.${PKG_VER}-linux-x86_64-cuda-cuda12.old.$$"
    mv "$DEST_PKG" "$OLD_DEST"
    mv "$TMP_DEST" "$DEST_PKG"
    rm -rf "$OLD_DEST"
else
    mv "$TMP_DEST" "$DEST_PKG"
fi
rm -rf "$BUILD_ABS/out-pkg"

# Clear any manual engine rollback pin so the newly built engine resolves as active
STATE_JSON="$COMPONENTS_DIR/.updates/state.json"
if [[ -f "$STATE_JSON" ]]; then
    python3 - "$STATE_JSON" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    if isinstance(data.get("engine"), dict) and data["engine"].get("pinned"):
        print(f"==> Clearing pinned engine ({data['engine']['pinned']}) in {p}")
        data["engine"]["pinned"] = None
        p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
except Exception:
    pass
PY
fi

echo "==> Staging and activating Control component $PKG_VER from $STRATA_REPO"
PYTHONDONTWRITEBYTECODE=1 python3 "$STRATA_REPO/packaging/stage_local_control.py" \
    --components "$COMPONENTS_DIR" \
    --version "$PKG_VER" \
    --commit "$GIT_FULL"

CONTROL_PY="$COMPONENTS_DIR/control/$PKG_VER/python/bin/python3"
echo "==> Running DiskSessionCache test suite in staged Control ($PKG_VER)"
(
    cd "$COMPONENTS_DIR/control/$PKG_VER"
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
        "$CONTROL_PY" -B -m unittest serve.test_disk_cache
)

if [[ "$RUN_TESTS" -eq 1 ]]; then
    echo "==> Running Control slot/session/bundle/isolation test suites"
    (
        cd "$COMPONENTS_DIR/control/$PKG_VER"
        PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
            "$CONTROL_PY" -B -m unittest serve.test_slots serve.test_session_save_retry
    )
    (
        cd "$STRATA_REPO"
        PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
            "$CONTROL_PY" -B -m unittest \
            tests/terminal/test_no_acquisition.py \
            tests/terminal/test_bundle_imports.py \
            tests/terminal/test_bundle_tools_paths.py \
            tests/terminal/test_handover.py
    )
fi

if [[ "$DO_PUSH" -eq 1 ]]; then
    BRANCH="$(git -C "$STRATA_REPO" rev-parse --abbrev-ref HEAD)"
    echo "==> Pushing $BRANCH to origin (--force-with-lease)"
    git -C "$STRATA_REPO" push --force-with-lease origin "$BRANCH"
fi

echo "==> Done!"
echo "    Engine  : $DEST_PKG"
echo "    Control : $COMPONENTS_DIR/control/$PKG_VER"
echo "    Restart Strata when ready (e.g. 'systemctl --user restart strata')."
