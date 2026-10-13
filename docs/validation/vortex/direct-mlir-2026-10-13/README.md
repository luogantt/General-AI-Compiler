# Direct BEVFormer MLIR → VOLT validation record

Date: 2026-10-13

This record covers the phase-1 device compiler path:

```text
BEVFormer FFN graph
  → LLVM-dialect MLIR
  → mlir-opt
  → mlir-translate --mlir-to-llvmir
  → VOLT clang/Vortex backend
  → RV32 Vortex relocatable object
  → Vortex startup + libvortex2 + ld.lld
  → ELF → vxbin.py → .vxbin
  → vortex2 host → SimX
```

It validates that the generated device code can reach the Vortex machine-code
backend without a generated C++ device kernel, link with the Vortex startup,
package as `.vxbin`, and execute all three FFN kernels in SimX with numerical
parity. It does not establish correctness or performance on Vortex hardware.

## Pinned toolchain

All compiler stages came from one Vortex LLVM tree:

| Item | Validated value |
|---|---|
| LLVM/VOLT revision | `b2ecfe086df2d00233a87f3d3ebe6a74aba129e2` |
| LLVM version | `20.1.8` |
| Vortex release / revision | `v3.0` / `7fe19369587c087f902c3e2bc2add612aa910a95` |
| Vortex toolchain bundle | `v3.0.1`; RV32 GCC `12.2.0` |
| SoftFloat revision | `b51ef8f3201669b2288104c28546fc72532a1ea4` |
| Ramulator revision | `e62c84a6f0e06566ba6e182d308434b4532068a5` |
| Projects | `clang;lld;mlir` |
| Targets | `RISCV;X86` |
| Build | Release with assertions |
| Device triple | `riscv32-unknown-unknown-elf` |
| Device ISA/features | `rv32imaf`, `+xvortex`, `+zicond` |
| Device ABI | `ilp32f` |

`mlir-opt`, `mlir-translate`, `opt`, `llc`, `clang`, `clang++`, and `ld.lld`
must remain at this revision. LLVM 24 bitcode or textual IR is not used as a
version boundary.

## Results recorded

The following checks passed with the pinned LLVM 20.1.8/VOLT build:

1. `mlir-opt --verify-each --canonicalize --cse` accepted the generated
   Vector Add and BEVFormer FFN LLVM-dialect modules.
2. `mlir-translate --mlir-to-llvmir` translated both optimized modules to
   LLVM IR.
3. `llc` compiled the Vector Add and FFN bring-up modules to 32-bit
   little-endian RISC-V relocatable objects using the single-float ABI.
   The generated CLI artifact was also compiled with the Vortex `common.mk`
   Clang driver flags, including `-Werror`, `+xvortex`, `+zicond`,
   `-march=rv32imaf`, and `-mabi=ilp32f`.
4. The Vector Add object contained both `vector_add` and
   `__vx_kentry_vector_add`.
5. The BEVFormer object contained each implementation and launch alias:
   - `bevformer_fc1_bias_relu` and
     `__vx_kentry_bevformer_fc1_bias_relu`
   - `bevformer_fc2_bias_residual` and
     `__vx_kentry_bevformer_fc2_bias_residual`
   - `bevformer_layer_norm` and
     `__vx_kentry_bevformer_layer_norm`
6. The objects contained retained executable entry sections. The recorded
   sections were `.vx_entry.vector_add`,
   `.vx_entry.bevformer_fc1_bias_relu`,
   `.vx_entry.bevformer_fc2_bias_residual`, and
   `.vx_entry.bevformer_layer_norm`, each with `AXR` flags.
7. Disassembly contained CTA CSR reads for `0xcd3` (thread id x), `0xcd6`
   (block id x), and `0xcd9` (block dimension x).
8. Disassembly contained VOLT-generated `vx_split_n` and `vx_join`
   instructions around divergent bounds checks. This confirms that the
   uniform markers and branch-divergence path reached Vortex instruction
   selection.
9. `ld.lld` linked the generated object with the Vortex KMU startup and
   `libvortex2.a`. The final ELF is an ELF32 little-endian RISC-V executable,
   entry `0x80000000`, with the single-float ABI and all three
   `__vx_kentry_*` symbols.
10. `vxbin.py` produced `kernel.vxbin`; its `VXSYMTAB` contains
    `bevformer_fc1_bias_relu`, `bevformer_fc2_bias_residual`, and
    `bevformer_layer_norm`.
11. The generated `vortex2` host loaded that `.vxbin` and completed a SimX
    warm-up followed by 10 measured runs. The result was finite and matched
    the frozen NumPy oracle with maximum absolute error `1.1920929e-7`, below
    the `1.0e-3` acceptance tolerance.

The final deterministic CLI smoke artifact recorded by this session had these
content hashes after the complete direct-MLIR/Vortex pipeline:

| File | SHA-256 |
|---|---|
| `kernel.mlir` | `7ad12f7bdfbff9cc3e29fe70978ef82edf43fb3b5979fe9a6fcd1716fcaf9a63` |
| `kernel.opt.mlir` | `7ad12f7bdfbff9cc3e29fe70978ef82edf43fb3b5979fe9a6fcd1716fcaf9a63` |
| `kernel.ll` | `44e52dd68f5b506b0a17876ad5d14d685bedda8fd9818a978619cf7862d9507c` |
| `kernel.o` | `d39b3be0e7c0c75ff6d7815a89d0529ca6d634672753f71c11a6911fa15beaa3` |
| `kernel.elf` | `88bead43bbaeed6dd9a91adebe0501db211987cef7b4f8b06ccf577796e4735d` |
| `kernel.vxbin` | `0ecb8582200acae7f89707b06e6f5afe40d45f72ea2f815464e1bddcf0ddae20` |
| `kernel.dump` | `be75b25fcad3f241acac85d429c4eb00b1960df6ca5e705e4feafd2761d7d34c` |
| `gac_bevformer_vortex` | `fd79043ce84aa6bde2e00c58da1a796edac1824e628ce1df52a2d49648a7b809` |

The translated module triple was
`riscv32-unknown-unknown-elf`. This is the canonical triple emitted by the
VOLT Clang driver for `--target=riscv32-unknown-elf`, so compiling LLVM IR with
the Vortex `-Werror` flags does not trigger `-Woverride-module`.

The raw repeat-10 result is preserved in [simx-repeat10.log](simx-repeat10.log),
and the output digests are preserved in
[artifact-hashes.sha256](artifact-hashes.sha256).
The log's `optimized_mlir_sha256` identifies the upstream Tensor/Linalg
`optimized.mlir` used to derive the call plan; it is distinct from the device
module's `kernel.opt.mlir` hash listed above.

The repeat-10 SimX means were `4.447282 ms` for the runtime event-profile,
`4.513678 ms` for host-observed kernel time, and `4.602381 ms` end to end.
These are software-simulator wall-clock measurements for the tiny synthetic
shape `[1,2,4] → 8 → 4`; they are not Vortex hardware latency or an inference
speed claim.

## Reproduction

Run these commands on Linux or WSL from a clean checkout. Paths are supplied
through variables so the procedure does not depend on a particular user home
directory.

Build the exact toolchain:

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export LLVM_ROOT=/path/to/vortex-llvm-project
export GAC_VOLT_CACHE_ROOT="$HOME/.cache/gac-volt-llvm20"

git -C "$LLVM_ROOT" checkout b2ecfe086df2d00233a87f3d3ebe6a74aba129e2
LLVM_ROOT="$LLVM_ROOT" \
GAC_VOLT_CACHE_ROOT="$GAC_VOLT_CACHE_ROOT" \
  bash "$REPO_ROOT/tools/build_volt_mlir_wsl.sh"

export LLVM_PREFIX="$GAC_VOLT_CACHE_ROOT/current-build"
test "$(cat "$LLVM_PREFIX/.gac-build-revision")" = \
  b2ecfe086df2d00233a87f3d3ebe6a74aba129e2
for tool in mlir-opt mlir-translate opt llc clang clang++ ld.lld \
            llvm-nm llvm-readelf llvm-objdump llvm-objcopy; do
  test -x "$LLVM_PREFIX/bin/$tool"
done
"$LLVM_PREFIX/bin/mlir-opt" --version | grep -F '20.1.8'
"$LLVM_PREFIX/bin/mlir-translate" --version | grep -F '20.1.8'
"$LLVM_PREFIX/bin/llc" --version | grep -F '20.1.8'
```

Generate a fresh deterministic phase-1 BEVFormer FFN artifact:

```bash
export ARTIFACT="${ARTIFACT:-$REPO_ROOT/build/direct-mlir-$(date +%Y%m%d-%H%M%S)}"

python3 "$REPO_ROOT/tools/build_bevformer_vortex.py" \
  --synthetic-smoke \
  --device-lowering direct-mlir \
  --mlir-opt "$LLVM_PREFIX/bin/mlir-opt" \
  --output "$ARTIFACT"

python3 "$ARTIFACT/verify_artifact.py"
test -f "$ARTIFACT/kernel.mlir"
test ! -e "$ARTIFACT/kernel.cpp"
```

Translate and compile the device module:

```bash
export VALIDATION_DIR="$ARTIFACT/object-validation"
mkdir -p "$VALIDATION_DIR"

"$LLVM_PREFIX/bin/mlir-opt" \
  --verify-each --canonicalize --cse \
  "$ARTIFACT/kernel.mlir" \
  -o "$VALIDATION_DIR/kernel.opt.mlir"

"$LLVM_PREFIX/bin/mlir-translate" \
  --mlir-to-llvmir \
  "$VALIDATION_DIR/kernel.opt.mlir" \
  -o "$VALIDATION_DIR/kernel.ll"

"$LLVM_PREFIX/bin/opt" \
  -passes=verify -disable-output \
  "$VALIDATION_DIR/kernel.ll"

"$LLVM_PREFIX/bin/clang++" \
  --target=riscv32-unknown-elf \
  -Xclang -target-feature -Xclang +xvortex \
  -Xclang -target-feature -Xclang +zicond \
  -mllvm -disable-loop-idiom-all \
  -Wno-unused-command-line-argument \
  -march=rv32imaf -mabi=ilp32f \
  -Wall -Wextra -Wfatal-errors -Werror \
  -O3 -mcmodel=medany \
  -fno-rtti -fno-exceptions -nostartfiles -nostdlib \
  -fdata-sections -ffunction-sections \
  -c "$VALIDATION_DIR/kernel.ll" \
  -o "$VALIDATION_DIR/kernel.o"
```

Check the object ABI and Vortex instruction selection:

```bash
"$LLVM_PREFIX/bin/llvm-readelf" -h "$VALIDATION_DIR/kernel.o"

"$LLVM_PREFIX/bin/llvm-readelf" -s -W "$VALIDATION_DIR/kernel.o" \
  | tee "$VALIDATION_DIR/symbols.txt"
for symbol in \
  __vx_kentry_bevformer_fc1_bias_relu \
  __vx_kentry_bevformer_fc2_bias_residual \
  __vx_kentry_bevformer_layer_norm; do
  grep -F "$symbol" "$VALIDATION_DIR/symbols.txt"
done

"$LLVM_PREFIX/bin/llvm-readelf" -S -W "$VALIDATION_DIR/kernel.o" \
  | tee "$VALIDATION_DIR/sections.txt"
grep -E '\.vx_entry\.bevformer_.*[[:space:]]AXR[[:space:]]' \
  "$VALIDATION_DIR/sections.txt"

"$LLVM_PREFIX/bin/llvm-objdump" -d \
  --mattr=+m,+a,+f,+xvortex,+zicond \
  "$VALIDATION_DIR/kernel.o" \
  | tee "$VALIDATION_DIR/kernel.dump"
grep -Ei 'csrr.*0xcd3' "$VALIDATION_DIR/kernel.dump"
grep -Ei 'csrr.*0xcd6' "$VALIDATION_DIR/kernel.dump"
grep -Ei 'csrr.*0xcd9' "$VALIDATION_DIR/kernel.dump"
grep -E 'vx_split(_n)?' "$VALIDATION_DIR/kernel.dump"
grep -F 'vx_join' "$VALIDATION_DIR/kernel.dump"
```

The Vector Add bring-up follows the same `mlir-opt`, `mlir-translate`, and
`llc` commands with the generated Vector Add `kernel.mlir`. Its object must
contain `vector_add`, `__vx_kentry_vector_add`, and
`.vx_entry.vector_add` before proceeding to link and runtime tests.

## Full Vortex build and SimX reproduction

Deploy into a fresh regression directory and run the generated Makefile with
the same `LLVM_PREFIX`:

```bash
export VORTEX_HOME=/path/to/vortex
export DEPLOY_DIR="$VORTEX_HOME/tests/regression/gac_bevformer_direct_mlir"
mkdir -p "$DEPLOY_DIR"
rsync -av --delete "$ARTIFACT/" "$DEPLOY_DIR/"

make -C "$DEPLOY_DIR" LLVM_PATH="$LLVM_PREFIX" all
"$LLVM_PREFIX/bin/llvm-readelf" -s -W "$DEPLOY_DIR/kernel.elf" \
  | grep __vx_kentry_bevformer
test -s "$DEPLOY_DIR/kernel.vxbin"
make -C "$DEPLOY_DIR" LLVM_PATH="$LLVM_PREFIX" run-simx \
  OPTS="--data-dir . --kernel kernel.vxbin --warmup 1 --repeat 10"
```

The software-runtime gates are complete: startup/runtime link,
`VXSYMTAB` packaging, and SimX numerical parity. Remaining target acceptance
requires repeating numerical validation on Vortex hardware and measuring
warm-up-separated hardware latency there.

No inference-speed or acceleration number is claimed by this record. The
current phase-1 kernels are scalar correctness kernels; tiling, LMEM reuse,
fusion, double buffering, and WMMA/WGMMA work remain separate performance
milestones.
