from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SM86_FIXTURE = REPOSITORY_ROOT / "examples" / "mlir_sm86_add_relu.mlir"
NVVM_PIPELINE = (
    "--gpu-lower-to-nvvm-pipeline="
    "kernel-bare-ptr-calling-convention=1 "
    "cubin-chip=sm_86 "
    "cubin-format=isa "
    "allow-pattern-rollback=0"
)


@unittest.skipUnless(
    os.environ.get("GAC_MLIR_OPT"),
    "set GAC_MLIR_OPT to run the optional SM86 MLIR/LLVM NVPTX smoke test",
)
class MlirSm86ToolchainIntegrationTests(unittest.TestCase):
    def test_gpu_dialect_lowers_to_embedded_sm86_ptx(self):
        mlir_opt = os.environ["GAC_MLIR_OPT"]
        with tempfile.TemporaryDirectory(prefix="gac-sm86-") as directory:
            output = Path(directory) / "sm86-binary.mlir"
            try:
                completed = subprocess.run(
                    [
                        mlir_opt,
                        str(SM86_FIXTURE),
                        NVVM_PIPELINE,
                        "-o",
                        str(output),
                    ],
                    shell=False,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=120,
                    check=False,
                )
            except OSError as exc:
                self.fail(f"failed to launch GAC_MLIR_OPT={mlir_opt!r}: {exc}")

            self.assertEqual(
                completed.returncode,
                0,
                f"SM86 MLIR lowering failed:\n{completed.stderr}",
            )
            self.assertTrue(output.is_file(), "mlir-opt did not create the output module")
            lowered = output.read_text(encoding="utf-8")

        self.assertIn("gpu.binary @gac_kernels", lowered)
        self.assertRegex(lowered, r'#nvvm\.target<chip = "sm_86">')
        self.assertIn(".target sm_86", lowered)
        self.assertRegex(lowered, r"\.visible \.entry add_relu_broadcast\(")
        self.assertNotIn("gpu.func @add_relu_broadcast", lowered)


if __name__ == "__main__":
    unittest.main()
