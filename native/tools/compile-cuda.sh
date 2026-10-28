#!/usr/bin/env bash
# ONNX semantics and target lowering live in the C++ MLIR compiler. This script
# only connects the official MLIR/LLVM tools and NVIDIA device math library.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'Usage: GAC_NATIVE_BIN=... LLVM_PATH=... CUDA_LIBDEVICE=... compile-cuda.sh model.onnx output-directory' >&2
  exit 2
fi
: "${GAC_NATIVE_BIN:?directory containing gac-compile}"
: "${LLVM_PATH:?LLVM/MLIR 20.1.8 build with the NVPTX backend enabled}"
gac_libdevice="${CUDA_LIBDEVICE:-${CUDA_HOME:-/usr/local/cuda}/nvvm/libdevice/libdevice.10.bc}"
gac_arch="${GAC_CUDA_ARCH:-sm_86}"
gac_ptx="${GAC_CUDA_PTX:-+ptx80}"
[[ "$gac_arch" =~ ^sm_[0-9]+[af]?$ ]] || { echo 'Invalid GAC_CUDA_ARCH' >&2; exit 2; }
[[ "$gac_ptx" =~ ^\+ptx[0-9]+$ ]] || { echo 'Invalid GAC_CUDA_PTX' >&2; exit 2; }
[[ -f "$gac_libdevice" ]] || { echo "CUDA libdevice not found: $gac_libdevice" >&2; exit 2; }
for gac_tool in mlir-translate llvm-link opt llc; do
  [[ -x "$LLVM_PATH/bin/$gac_tool" ]] || { echo "Missing LLVM tool: $gac_tool" >&2; exit 2; }
  "$LLVM_PATH/bin/$gac_tool" --version | grep -q 'version 20\.1\.8' || {
    echo "$gac_tool must match the compiler's LLVM 20.1.8" >&2; exit 2;
  }
done
"$LLVM_PATH/bin/llc" --version | grep -q nvptx64 || {
  echo 'LLVM was built without NVPTX; enable LLVM_TARGETS_TO_BUILD=...;NVPTX' >&2
  exit 2
}
gac_model="$(realpath -- "$1")"
gac_output="$(realpath -m -- "$2")"
mkdir -p -- "$gac_output"
# Every invocation owns a fresh artifact directory. A failed build must not
# leave an older PTX/cubin looking like the result of the current manifest.
gac_existing="$(find "$gac_output" -mindepth 1 -maxdepth 1 -print -quit)"
if [[ -n "$gac_existing" ]]; then
  echo "Output directory must be new or empty: $gac_output" >&2
  exit 2
fi
gac_compiler="$(realpath -- "$GAC_NATIVE_BIN/gac-compile")"
# Capture the executable before running it: hashing only after compilation can
# falsely attribute a running old executable to a newly rebuilt path.
gac_compiler_identity="$(sha256sum -- "$gac_compiler")"
printf '%s\n' "$gac_compiler_identity" > "$gac_output/compiler-before.sha256"
gac_check_compiler() {
  if [[ "$(sha256sum -- "$gac_compiler")" != "$gac_compiler_identity" ]]; then
    echo 'Compiler changed during artifact generation; retry using a fixed compiler copy.' >&2
    exit 1
  fi
}
"$gac_compiler" "$gac_model" --emit=llvm --target=cuda \
  --regular-lowering "${GAC_REGULAR_LOWERING:-linalg}" \
  --cuda-matmul-schedule "${GAC_CUDA_MATMUL_SCHEDULE:-scalar}" \
  --cuda-tile-m "${GAC_CUDA_TILE_M:-32}" --cuda-tile-n "${GAC_CUDA_TILE_N:-32}" \
  --cuda-tile-k "${GAC_CUDA_TILE_K:-16}" \
  --cuda-thread-tile-m "${GAC_CUDA_THREAD_TILE_M:-2}" \
  --cuda-thread-tile-n "${GAC_CUDA_THREAD_TILE_N:-2}" \
  --cuda-reduction-schedule "${GAC_CUDA_REDUCTION_SCHEDULE:-scalar}" \
  --cuda-conv-channel-tile "${GAC_CUDA_CONV_CHANNEL_TILE:-1}" \
  --constant-fold-profile "${GAC_CONSTANT_FOLD_PROFILE:-default}" \
  --save-stages "$gac_output" --manifest "$gac_output/plan.json" \
  --constants-dir "$gac_output" -o "$gac_output/kernel.mlir"
gac_check_compiler
"$LLVM_PATH/bin/mlir-translate" --mlir-to-llvmir "$gac_output/kernel.mlir" \
  -o "$gac_output/kernel.ll"
# Keep graph entries externally visible; internalize only symbols imported
# from libdevice. Link only the required functions, then eliminate dead code.
"$LLVM_PATH/bin/llvm-link" --only-needed --internalize \
  "$gac_output/kernel.ll" "$gac_libdevice" -o "$gac_output/kernel-linked.bc"
"$LLVM_PATH/bin/opt" -passes='default<O2>' \
  "$gac_output/kernel-linked.bc" -o "$gac_output/kernel-optimized.bc"
"$LLVM_PATH/bin/llc" -march=nvptx64 -mcpu="$gac_arch" -mattr="$gac_ptx" \
  -O2 -fp-contract=off -nvptx-fma-level=0 \
  "$gac_output/kernel-optimized.bc" -o "$gac_output/kernel.ptx"
# Offline assembly is optional. It validates PTX for the chosen GPU without a
# physical GPU, but cannot establish numerical correctness or inference FPS.
gac_ptxas="${CUDA_PTXAS:-}"
if [[ -n "$gac_ptxas" ]]; then
  "$gac_ptxas" --gpu-name "$gac_arch" --fmad=false \
    "$gac_output/kernel.ptx" -o "$gac_output/kernel.cubin"
fi
gac_check_compiler
{
  printf 'target=%s\nptx_features=%s\n' "$gac_arch" "$gac_ptx"
  "$LLVM_PATH/bin/llc" --version
  printf '%s\n' "$gac_compiler_identity"
  sha256sum -- "$gac_libdevice" \
    "$gac_output/plan.json" "$gac_output/kernel.ptx"
} > "$gac_output/compile-provenance.txt"
echo "Native MLIR CUDA artifact: $gac_output/kernel.ptx"
