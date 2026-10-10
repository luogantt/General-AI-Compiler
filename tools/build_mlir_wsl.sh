#!/usr/bin/env bash
# Build the small LLVM/MLIR tool set used by GAC without modifying llvm-project.
set -euo pipefail

LLVM_ROOT="${LLVM_ROOT:-/mnt/c/Users/Thomas/Desktop/universal_ai_compile/llvm-project}"
GAC_MLIR_CACHE_ROOT="${GAC_MLIR_CACHE_ROOT:-$HOME/.cache/gac-mlir}"
JOBS="${JOBS:-4}"

for command in git tar cmake gcc g++ make realpath sed grep find ln; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "required build command is missing: $command" >&2
    exit 1
  fi
done

if [[ ! -d "$LLVM_ROOT/.git" ]]; then
  echo "llvm-project checkout not found: $LLVM_ROOT" >&2
  exit 1
fi

LLVM_REVISION="$(git -C "$LLVM_ROOT" rev-parse HEAD)"
GAC_MLIR_CACHE_ROOT="$(realpath -m "$GAC_MLIR_CACHE_ROOT")"
case "$GAC_MLIR_CACHE_ROOT" in
  "$HOME"/*) ;;
  *)
    echo "GAC_MLIR_CACHE_ROOT must resolve inside HOME: $GAC_MLIR_CACHE_ROOT" >&2
    exit 1
    ;;
esac

cache_marker="$GAC_MLIR_CACHE_ROOT/.gac-mlir-cache-root"
if [[ -d "$GAC_MLIR_CACHE_ROOT" ]] &&
   [[ -n "$(find "$GAC_MLIR_CACHE_ROOT" -mindepth 1 -maxdepth 1 -print -quit)" ]] &&
   [[ ! -f "$cache_marker" ]]; then
  echo "refusing to use a non-empty, unmanaged cache root: $GAC_MLIR_CACHE_ROOT" >&2
  exit 1
fi
mkdir -p "$GAC_MLIR_CACHE_ROOT"
printf '%s\n' 'managed by General-AI-Compiler tools/build_mlir_wsl.sh' >"$cache_marker"

GAC_MLIR_SOURCE="${GAC_MLIR_SOURCE:-$GAC_MLIR_CACHE_ROOT/sources/$LLVM_REVISION}"
GAC_MLIR_BUILD="${GAC_MLIR_BUILD:-$GAC_MLIR_CACHE_ROOT/builds/$LLVM_REVISION}"
GAC_MLIR_SOURCE="$(realpath -m "$GAC_MLIR_SOURCE")"
GAC_MLIR_BUILD="$(realpath -m "$GAC_MLIR_BUILD")"
for managed_path in "$GAC_MLIR_SOURCE" "$GAC_MLIR_BUILD"; do
  case "$managed_path" in
    "$GAC_MLIR_CACHE_ROOT"/*) ;;
    *)
      echo "source and build paths must stay below $GAC_MLIR_CACHE_ROOT: $managed_path" >&2
      exit 1
      ;;
  esac
done
case "$GAC_MLIR_BUILD/" in
  "$GAC_MLIR_SOURCE/"*)
    echo "GAC_MLIR_BUILD must not be inside GAC_MLIR_SOURCE" >&2
    exit 1
    ;;
esac
case "$GAC_MLIR_SOURCE/" in
  "$GAC_MLIR_BUILD/"*)
    echo "GAC_MLIR_SOURCE must not be inside GAC_MLIR_BUILD" >&2
    exit 1
    ;;
esac

# A Windows checkout may contain CRLF shell helpers that CMake executes while
# configuring LLVM.  A git archive gives WSL a clean LF source snapshot and
# leaves the user's checkout untouched.  Revision-specific directories avoid
# deleting or replacing any caller-provided directory when LLVM changes.
source_marker="$GAC_MLIR_SOURCE/.gac-source-revision"
if [[ ! -d "$GAC_MLIR_SOURCE" ]]; then
  mkdir -p "$GAC_MLIR_SOURCE"
  git -C "$LLVM_ROOT" archive HEAD \
    llvm mlir cmake third-party libc LICENSE.TXT | tar -x -C "$GAC_MLIR_SOURCE"
  printf '%s\n' "$LLVM_REVISION" >"$source_marker"
elif [[ ! -f "$source_marker" ]]; then
  echo "refusing to reuse an unmanaged source directory: $GAC_MLIR_SOURCE" >&2
  exit 1
elif [[ "$(cat "$source_marker")" != "$LLVM_REVISION" ]]; then
  echo "source cache revision does not match $LLVM_REVISION: $GAC_MLIR_SOURCE" >&2
  exit 1
fi

build_marker="$GAC_MLIR_BUILD/.gac-build-revision"
if [[ -d "$GAC_MLIR_BUILD" ]] && [[ ! -f "$build_marker" ]]; then
  echo "refusing to reuse an unmanaged build directory: $GAC_MLIR_BUILD" >&2
  exit 1
elif [[ -f "$build_marker" ]] &&
     [[ "$(cat "$build_marker")" != "$LLVM_REVISION" ]]; then
  echo "refusing to relabel a build from another revision: $GAC_MLIR_BUILD" >&2
  exit 1
fi
mkdir -p "$GAC_MLIR_BUILD"
printf '%s\n' "$LLVM_REVISION" >"$build_marker"

cmake -S "$GAC_MLIR_SOURCE/llvm" -B "$GAC_MLIR_BUILD" \
  -G "Unix Makefiles" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DMLIR_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_DOCS=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF

cmake --build "$GAC_MLIR_BUILD" \
  --target mlir-opt mlir-translate llc \
  --parallel "$JOBS"

"$GAC_MLIR_BUILD/bin/mlir-opt" --version
"$GAC_MLIR_BUILD/bin/mlir-translate" --version
"$GAC_MLIR_BUILD/bin/llc" --version | sed -n '1,12p'
"$GAC_MLIR_BUILD/bin/llc" --version | grep -E 'nvptx(64)?'

current_build="$GAC_MLIR_CACHE_ROOT/current-build"
if [[ -e "$current_build" ]] && [[ ! -L "$current_build" ]]; then
  echo "refusing to replace a non-symlink current-build path: $current_build" >&2
  exit 1
fi
ln -sfn "$GAC_MLIR_BUILD" "$current_build"

cat <<EOF

GAC MLIR tools are ready:
  LLVM revision: $LLVM_REVISION
  $GAC_MLIR_BUILD/bin/mlir-opt
  $GAC_MLIR_BUILD/bin/mlir-translate
  $GAC_MLIR_BUILD/bin/llc
  Stable tool link: $current_build
EOF
