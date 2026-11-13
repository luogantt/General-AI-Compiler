#!/usr/bin/env python3
"""Check the fail-closed command-line boundary for Ascend TTIR export."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import tempfile


CONTRACT_ERROR = "standard-TTIR export only"


def run(compiler: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(compiler), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=60,
        check=False,
    )


def expect_rejected(compiler: Path, arguments: list[str], expected: str) -> None:
    result = run(compiler, *arguments)
    assert result.returncode != 0, (arguments, result.stdout)
    assert expected in result.stdout, (arguments, expected, result.stdout)


def write_mlp_onnx(path: Path) -> None:
    """Write a tiny, selector-compatible form of the real six-op G0.5 slice."""

    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        raise RuntimeError(
            "the Ascend end-to-end contract requires the Python onnx and numpy packages"
        ) from exc

    hidden, intermediate, tokens = 16, 32, 16
    gate = (np.arange(hidden * intermediate, dtype=np.float32) / 4096).reshape(
        hidden, intermediate
    )
    up = (np.arange(hidden * intermediate, dtype=np.float32) / 8192).reshape(
        hidden, intermediate
    )
    down = (np.arange(intermediate * hidden, dtype=np.float32) / 4096).reshape(
        intermediate, hidden
    )
    nodes = [
        helper.make_node("MatMul", ["hidden_states", "gate_proj_weight"], ["gate"], name="gate_proj"),
        helper.make_node("Sigmoid", ["gate"], ["gate_sigmoid"], name="gate_sigmoid"),
        helper.make_node("Mul", ["gate", "gate_sigmoid"], ["gate_activated"], name="gate_silu"),
        helper.make_node("MatMul", ["hidden_states", "up_proj_weight"], ["up"], name="up_proj"),
        helper.make_node("Mul", ["gate_activated", "up"], ["gated_up"], name="swiglu_mul"),
        helper.make_node("MatMul", ["gated_up", "down_proj_weight"], ["output"], name="down_proj"),
    ]
    graph = helper.make_graph(
        nodes,
        "galaxeavla_qwen35_action_expert_layer0_mlp_contract",
        [helper.make_tensor_value_info("hidden_states", TensorProto.FLOAT, [1, tokens, hidden])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, tokens, hidden])],
        initializer=[
            numpy_helper.from_array(gate, name="gate_proj_weight"),
            numpy_helper.from_array(up, name="up_proj_weight"),
            numpy_helper.from_array(down, name="down_proj_weight"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="GAC Ascend contract",
        opset_imports=[helper.make_operatorsetid("", 17)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    onnx.save_model(model, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    assert compiler.is_file(), compiler

    with tempfile.TemporaryDirectory(prefix="gac-ascend-cli-") as temporary:
        root = Path(temporary)
        missing_model = root / "missing.onnx"
        manifest = root / "plan.json"
        artifacts = root / "ttir"
        constants = root / "constants"
        common = [
            str(missing_model),
            "--target=ascend",
            "--regular-lowering=linalg",
        ]

        expect_rejected(
            compiler,
            [
                *common,
                "--emit=device",
                f"--manifest={manifest}",
                f"--constants-dir={constants}",
            ],
            CONTRACT_ERROR,
        )
        expect_rejected(
            compiler,
            [
                *common,
                "--emit=llvm",
                f"--manifest={manifest}",
                f"--triton-dir={artifacts}",
                f"--constants-dir={constants}",
            ],
            "Ascend LLVM/binary/runtime lowering is not available",
        )
        expect_rejected(
            compiler,
            [
                *common,
                "--emit=device",
                f"--triton-dir={artifacts}",
                f"--constants-dir={constants}",
            ],
            CONTRACT_ERROR,
        )
        expect_rejected(
            compiler,
            [
                *common,
                "--emit=device",
                f"--manifest={manifest}",
                f"--triton-dir={artifacts}",
            ],
            CONTRACT_ERROR,
        )
        expect_rejected(
            compiler,
            [
                *common,
                "--emit=device",
                f"--manifest={manifest}",
                f"--vortex-triton-dir={artifacts}",
                f"--constants-dir={constants}",
            ],
            "Vortex-only compatibility option",
        )
        expect_rejected(
            compiler,
            [
                *common,
                "--emit=device",
                f"--manifest={manifest}",
                f"--triton-dir={artifacts}",
                f"--constants-dir={constants}",
                "--triton-num-warps=1",
            ],
            "backend launch option",
        )

        # The one accepted option shape must pass the CLI contract checks.  The
        # deliberately missing model then fails in the importer before any
        # TTIR directory, manifest, binary, or runtime artifact can be made.
        accepted = run(
            compiler,
            *common,
            "--emit=device",
            f"--manifest={manifest}",
            f"--triton-dir={artifacts}",
            f"--constants-dir={constants}",
        )
        assert accepted.returncode != 0, accepted.stdout
        assert CONTRACT_ERROR not in accepted.stdout, accepted.stdout
        assert "Ascend LLVM/binary/runtime lowering" not in accepted.stdout, accepted.stdout
        assert not manifest.exists() and not artifacts.exists(), accepted.stdout
        assert not constants.exists(), accepted.stdout

        # Exercise the actual model boundary end to end: ONNX importer,
        # GAC-to-Linalg legalization, all-Triton preflight, constants and TTIR.
        model = root / "galaxeavla-mlp.onnx"
        write_mlp_onnx(model)
        mlp_manifest = root / "mlp-plan.json"
        mlp_artifacts = root / "mlp-ttir"
        mlp_constants = root / "mlp-constants"
        mlp_stages = root / "mlp-stages"
        mlp_device = root / "mlp-device.mlir"
        compiled = run(
            compiler,
            str(model),
            "--target=ascend",
            "--regular-lowering=linalg",
            "--emit=device",
            f"--manifest={mlp_manifest}",
            f"--triton-dir={mlp_artifacts}",
            f"--constants-dir={mlp_constants}",
            f"--save-stages={mlp_stages}",
            f"-o={mlp_device}",
        )
        assert compiled.returncode == 0, compiled.stdout
        plan = json.loads(mlp_manifest.read_text(encoding="utf-8"))
        dispatches = plan["dispatches"]
        kinds = Counter(row["triton"]["kind"] for row in dispatches)
        assert plan["target"] == "ascend" and plan["execution"] == "artifact-only", plan
        assert plan["artifact_format"] == "standard-ttir", plan
        assert plan["runtime_available"] is False, plan
        assert "lmem_bytes" not in plan and "double_buffer" not in plan, plan
        assert plan["triton_dispatches"] == len(dispatches), plan
        assert kinds == {"matmul": 3, "pointwise": 1}, kinds
        assert plan["linalg_fused_producers"] >= 2, plan
        assert all(row["lowering"] == "linalg-to-triton" for row in dispatches), plan
        assert all(
            row["compute_template"]["id"].startswith("triton.standard.")
            and "triton_exclusion" not in row
            and "vortex" not in row
            for row in dispatches
        ), plan
        for row in dispatches:
            contract = row["triton"]
            assert contract["artifact_format"] == "standard-ttir", row
            assert contract["ttir_signature"], row
            assert not {
                "abi",
                "lanes",
                "block_size",
                "num_warps",
            }.intersection(contract), row
        linalg = (mlp_stages / "02-linalg.mlir").read_text(encoding="utf-8")
        assert linalg.count('gac.lowered_from = "gac.matmul"') == 3, linalg
        assert linalg.count('gac.lowered_from = "gac.sigmoid"') == 1, linalg
        assert linalg.count('gac.lowered_from = "gac.mul"') == 2, linalg
        ttir_files = sorted(mlp_artifacts.glob("*.ttir"))
        assert len(ttir_files) == plan["triton_dispatches"], ttir_files
        ttir_text = "\n".join(item.read_text(encoding="utf-8") for item in ttir_files)
        assert '"math.exp"' in ttir_text, ttir_text
        assert "vortex" not in ttir_text and "npubin" not in ttir_text, ttir_text
        assert len(list(mlp_constants.glob("tensor_*.bin"))) == 3, plan
        device_text = mlp_device.read_text(encoding="utf-8")
        assert "llvm.func" not in device_text and "func.func" not in device_text, device_text

    print(
        "passed: Ascend CLI is fail-closed and the complete six-op "
        "GalaxeaVLA MLP exports as all-Triton artifacts"
    )


if __name__ == "__main__":
    main()
