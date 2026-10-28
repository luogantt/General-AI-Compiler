#!/usr/bin/env bash
# All semantic compilation happens in gac-compile (C++ MLIR). Python is used
# only by the upstream Vortex configuration / ELF packaging utilities.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo 'Usage: GAC_NATIVE_BIN=... LLVM_PATH=... VORTEX_BUILD=... compile-vortex.sh model.onnx output-directory' >&2
  exit 2
fi
: "${GAC_NATIVE_BIN:?directory containing gac-compile and gac-opt}"
: "${LLVM_PATH:?the same VOLT LLVM/MLIR 20.1.8 build used for GAC}"
: "${VORTEX_BUILD:?configured Vortex RV32 build directory}"
gac_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
gac_model="$(realpath -- "$1")"
gac_output="$(realpath -m -- "$2")"
mkdir -p -- "$gac_output"
"$GAC_NATIVE_BIN/gac-compile" "$gac_model" --emit=llvm --target=vortex \
  --regular-lowering "${GAC_REGULAR_LOWERING:-linalg}" \
  --constant-fold-profile "${GAC_CONSTANT_FOLD_PROFILE:-default}" \
  --vortex-matmul-schedule "${GAC_VORTEX_MATMUL_SCHEDULE:-scalar}" \
  --tile-m "${GAC_TILE_M:-4}" --tile-n "${GAC_TILE_N:-4}" \
  --tile-k "${GAC_TILE_K:-16}" --vortex-lanes "${GAC_VORTEX_LANES:-4}" \
  --vortex-thread-tile-m "${GAC_VORTEX_THREAD_TILE_M:-1}" \
  --vortex-thread-tile-n "${GAC_VORTEX_THREAD_TILE_N:-1}" \
  --vortex-tcu-combined="${GAC_VORTEX_TCU_COMBINED:-0}" \
  --lmem-budget "${GAC_LMEM_BUDGET:-16384}" \
  --save-stages "$gac_output" --manifest "$gac_output/plan.json" \
  --constants-dir "$gac_output" -o "$gac_output/kernel.mlir"
make -C "$gac_output" -f "$gac_script_dir/Vortex.mk" \
  ROOT_DIR="$VORTEX_BUILD" LLVM_PATH="$LLVM_PATH" kernel.vxbin
# Disassemble executable sections only. -D would decode embedded tensor tables
# as instructions and produce misleading, very large reports.
"$LLVM_PATH/bin/llvm-objdump" -d "$gac_output/kernel.elf" > "$gac_output/kernel.dump"
echo "Native MLIR artifact: $gac_output/kernel.vxbin"
