#!/usr/bin/env bash
# Build one pinned LLVM 20.1.8 toolchain for MLIR -> VOLT -> Vortex.
set -euo pipefail

EXPECTED_REVISION="b2ecfe086df2d00233a87f3d3ebe6a74aba129e2"
GAC_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
LLVM_ROOT="${LLVM_ROOT:-$GAC_SCRIPT_DIR/../../llvm-project}"
GAC_VOLT_CACHE_ROOT="${GAC_VOLT_CACHE_ROOT:-$HOME/.cache/gac-volt-llvm20}"
JOBS="${JOBS:-6}"

for command in git tar cmake gcc g++ make realpath sed grep find ln; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required build command is missing: $command" >&2
    exit 1
  fi
done

LLVM_ROOT="$(realpath -m "$LLVM_ROOT")"
if [[ ! -d "$LLVM_ROOT/.git" ]]; then
  echo "VOLT llvm-project checkout not found: $LLVM_ROOT" >&2
  exit 1
fi

LLVM_REVISION="$(git -C "$LLVM_ROOT" rev-parse HEAD)"
if [[ "$LLVM_REVISION" != "$EXPECTED_REVISION" ]]; then
  echo "VOLT checkout must be pinned to $EXPECTED_REVISION; got $LLVM_REVISION" >&2
  exit 1
fi

GAC_VOLT_CACHE_ROOT="$(realpath -m "$GAC_VOLT_CACHE_ROOT")"
case "$GAC_VOLT_CACHE_ROOT" in
  "$HOME"/*) ;;
  *)
    echo "GAC_VOLT_CACHE_ROOT must resolve inside HOME: $GAC_VOLT_CACHE_ROOT" >&2
    exit 1
    ;;
esac

cache_marker="$GAC_VOLT_CACHE_ROOT/.gac-volt-cache-root"
if [[ -d "$GAC_VOLT_CACHE_ROOT" ]] &&
   [[ -n "$(find "$GAC_VOLT_CACHE_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]] &&
   [[ ! -f "$cache_marker" ]]; then
  echo "refusing to use a non-empty, unmanaged cache root: $GAC_VOLT_CACHE_ROOT" >&2
  exit 1
fi
mkdir -p "$GAC_VOLT_CACHE_ROOT"
printf '%s\n' 'managed by General-AI-Compiler tools/build_volt_mlir_wsl.sh' >"$cache_marker"

VOLT_SOURCE="${VOLT_SOURCE:-$GAC_VOLT_CACHE_ROOT/sources/$LLVM_REVISION}"
VOLT_BUILD="${VOLT_BUILD:-$GAC_VOLT_CACHE_ROOT/builds/$LLVM_REVISION}"
VOLT_SOURCE="$(realpath -m "$VOLT_SOURCE")"
VOLT_BUILD="$(realpath -m "$VOLT_BUILD")"
for managed_path in "$VOLT_SOURCE" "$VOLT_BUILD"; do
  case "$managed_path" in
    "$GAC_VOLT_CACHE_ROOT"/*) ;;
    *)
      echo "source and build paths must stay below $GAC_VOLT_CACHE_ROOT" >&2
      exit 1
      ;;
  esac
done

source_marker="$VOLT_SOURCE/.gac-source-revision"
if [[ ! -d "$VOLT_SOURCE" ]]; then
  mkdir -p "$VOLT_SOURCE"
  # git archive reads repository blobs and therefore avoids CRLF files from a
  # Windows worktree while keeping this build at the exact reviewed revision.
  git -C "$LLVM_ROOT" archive HEAD \
    llvm mlir clang lld cmake third-party compiler-rt libc LICENSE.TXT \
    | tar -x -C "$VOLT_SOURCE"
  printf '%s\n' "$LLVM_REVISION" >"$source_marker"
elif [[ ! -f "$source_marker" ]] || [[ "$(cat "$source_marker")" != "$LLVM_REVISION" ]]; then
  echo "refusing to reuse an unmanaged or mismatched source directory: $VOLT_SOURCE" >&2
  exit 1
fi

build_owner_marker="$VOLT_BUILD/.gac-managed-build-revision"
build_marker="$VOLT_BUILD/.gac-build-revision"
if [[ -d "$VOLT_BUILD" ]] &&
   [[ -n "$(find "$VOLT_BUILD" -mindepth 1 -maxdepth 1 -print -quit)" ]] &&
   { [[ ! -f "$build_owner_marker" ]] ||
     [[ "$(cat "$build_owner_marker")" != "$LLVM_REVISION" ]]; }; then
  echo "refusing to reuse an unmanaged or mismatched build directory: $VOLT_BUILD" >&2
  exit 1
fi
if [[ -f "$build_marker" ]] && [[ "$(cat "$build_marker")" != "$LLVM_REVISION" ]]; then
  echo "refusing to relabel a completed build from another revision: $VOLT_BUILD" >&2
  exit 1
fi
mkdir -p "$VOLT_BUILD"
printf '%s\n' "$LLVM_REVISION" >"$build_owner_marker"

cmake -S "$VOLT_SOURCE/llvm" -B "$VOLT_BUILD" \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
  -DLLVM_TARGETS_TO_BUILD="RISCV;X86" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DMLIR_INCLUDE_TESTS=OFF \
  -DCLANG_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_DOCS=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF

cmake --build "$VOLT_BUILD" \
  --target mlir-opt mlir-translate opt clang lld llc \
           llvm-nm llvm-objdump llvm-readelf llvm-objcopy \
  --parallel "$JOBS"

"$VOLT_BUILD/bin/mlir-opt" --version | grep -F '20.1.8'
"$VOLT_BUILD/bin/mlir-translate" --version | grep -F '20.1.8'
"$VOLT_BUILD/bin/clang" --version | grep -F '20.1.8'
"$VOLT_BUILD/bin/clang++" --version | grep -F '20.1.8'
"$VOLT_BUILD/bin/ld.lld" --version | grep -F '20.1.8'
"$VOLT_BUILD/bin/llc" --version | grep -E 'riscv(32|64)'

# This is a completion marker, not merely a configuration marker.  The direct
# backend and generated Makefile reject a toolchain until all required tools
# above have built successfully and passed their version probes.
printf '%s\n' "$LLVM_REVISION" >"$build_marker"

current_build="$GAC_VOLT_CACHE_ROOT/current-build"
if [[ -e "$current_build" ]] && [[ ! -L "$current_build" ]]; then
  echo "refusing to replace a non-symlink current-build path: $current_build" >&2
  exit 1
fi
ln -sfn "$VOLT_BUILD" "$current_build"

cat <<EOF

Pinned GAC VOLT/MLIR tools are ready:
  LLVM revision: $LLVM_REVISION
  Build root: $VOLT_BUILD
  mlir-opt: $VOLT_BUILD/bin/mlir-opt
  mlir-translate: $VOLT_BUILD/bin/mlir-translate
  clang: $VOLT_BUILD/bin/clang
  Stable tool link: $current_build
EOF
