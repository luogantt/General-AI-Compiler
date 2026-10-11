from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gac.backends.mlir_cublas_ptx import (
    PTX_ENTRY_SYMBOLS,
    compile_mlir_cublas_ptx,
)
from gac.bevformer_ffn import (
    BEVFormerFFNConfig,
    build_bevformer_ffn_graph,
    ffn_weight_shapes,
)
from gac.errors import BackendError
from gac.mlir_cublas_ptx_artifact import (
    ARTIFACT_FILES,
    MANIFEST_FILE,
    load_mlir_cublas_ptx_artifact,
    write_mlir_cublas_ptx_artifact,
)
from gac.mlir_toolchain import MlirPtxArtifact


def _graph():
    config = BEVFormerFFNConfig()
    weights = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in ffn_weight_shapes(config).items()
    }
    weights["norm_weight"].fill(1.0)
    return build_bevformer_ffn_graph(
        weights, config, name="artifact_round_trip_ffn"
    )


def _mlir_string(value: str) -> str:
    return "".join(f"\\{byte:02X}" for byte in value.encode("utf-8"))


def _fake_lowering(source: str, **_kwargs) -> MlirPtxArtifact:
    entries = []
    for symbol, count in zip(PTX_ENTRY_SYMBOLS, (2, 6), strict=True):
        parameters = ",\n".join(
            f"  .param .u64 .ptr .global {symbol}_param_{index}"
            for index in range(count)
        )
        entries.append(
            f".visible .entry {symbol}(\n{parameters}\n)\n"
            ".maxntid 256, 1, 1\n{ ret; }"
        )
    ptx = (
        ".version 7.1\n.target sm_86\n.address_size 64\n"
        + "\n".join(entries)
        + "\n"
    )
    binary_mlir = (
        "module {\n"
        "  gpu.binary @gac_bevformer_ffn_fusions_sm86 "
        "[#gpu.object<#nvvm.target<chip = \"sm_86\">, "
        f"properties = {{assembly = \"{_mlir_string(ptx)}\"}}, \"PTX\">]\n"
        "}\n"
    )
    return MlirPtxArtifact(
        binary_mlir=binary_mlir,
        ptx=ptx,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        command=("mlir-opt", "--gpu-lower-to-nvvm-pipeline=..."),
        mlir_opt_version="LLVM 24.0.0git (Optimized build)",
        mlir_opt_source_revision="a" * 40,
        target_chip="sm_86",
        target_features="+ptx71",
        ptx_version="7.1",
        entries=PTX_ENTRY_SYMBOLS,
    )


def _rewrite_manifest(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    unsigned = dict(value)
    unsigned.pop("manifest_payload_sha256", None)
    value["manifest_payload_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class MlirCublasPtxArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.graph = _graph()
        with patch(
            "gac.mlir_toolchain.lower_gpu_mlir_to_ptx",
            side_effect=_fake_lowering,
        ):
            cls.program = compile_mlir_cublas_ptx(
                cls.graph, command=("mock-mlir-opt",)
            )

    def test_atomic_weight_free_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "deployment"
            manifest = write_mlir_cublas_ptx_artifact(
                destination, self.program
            )
            self.assertEqual(
                {entry.name for entry in destination.iterdir()},
                {MANIFEST_FILE, *ARTIFACT_FILES.values()},
            )
            self.assertEqual(set(manifest["files"]), set(ARTIFACT_FILES))
            self.assertNotIn("tensor_source", manifest)
            self.assertFalse(any("weight" in name for name in ARTIFACT_FILES.values()))
            rebuilt = load_mlir_cublas_ptx_artifact(destination, self.graph)
            self.assertEqual(rebuilt.integrity_sha256, self.program.integrity_sha256)
            self.assertEqual(rebuilt.calls, self.program.calls)

    def test_graph_and_call_plan_are_reconstructed_and_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "deployment"
            write_mlir_cublas_ptx_artifact(destination, self.program)

            different = self.graph.clone()
            different.constants["fc1_bias"][0] += np.float32(1.0)
            with self.assertRaisesRegex(BackendError, "not bound to this graph"):
                load_mlir_cublas_ptx_artifact(destination, different)

            manifest_path = destination / MANIFEST_FILE
            _rewrite_manifest(
                manifest_path,
                lambda value: value["calls"][0].update({"m": 2499}),
            )
            with self.assertRaisesRegex(BackendError, "graph and call plan"):
                load_mlir_cublas_ptx_artifact(destination, self.graph)

    def test_rejects_tampered_member_extra_member_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "deployment"
            write_mlir_cublas_ptx_artifact(destination, self.program)
            ptx_path = destination / ARTIFACT_FILES["ptx"]
            ptx_path.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(BackendError, "(?:size|SHA-256) does not match"):
                load_mlir_cublas_ptx_artifact(destination, self.graph)

            destination = root / "with-extra"
            write_mlir_cublas_ptx_artifact(destination, self.program)
            (destination / "weights.bin").write_bytes(b"must not be accepted")
            with self.assertRaisesRegex(BackendError, "unexpected file set"):
                load_mlir_cublas_ptx_artifact(destination, self.graph)

            destination = root / "with-link"
            write_mlir_cublas_ptx_artifact(destination, self.program)
            member = destination / ARTIFACT_FILES["ptx"]
            target = root / "outside.ptx"
            target.write_bytes(member.read_bytes())
            member.unlink()
            try:
                member.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symbolic links are unavailable: {exc}")
            with self.assertRaisesRegex(BackendError, "symlink"):
                load_mlir_cublas_ptx_artifact(destination, self.graph)

    def test_rejects_duplicate_manifest_keys_and_payload_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "duplicate"
            write_mlir_cublas_ptx_artifact(destination, self.program)
            manifest_path = destination / MANIFEST_FILE
            manifest_path.write_text(
                '{"artifact":"one","artifact":"two"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(BackendError, "duplicate JSON key"):
                load_mlir_cublas_ptx_artifact(destination, self.graph)

            destination = root / "payload"
            write_mlir_cublas_ptx_artifact(destination, self.program)
            value = json.loads((destination / MANIFEST_FILE).read_text())
            value["dispatch_count"] = 5
            (destination / MANIFEST_FILE).write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(BackendError, "payload SHA-256 does not match"):
                load_mlir_cublas_ptx_artifact(destination, self.graph)

    def test_manifest_comparison_preserves_json_scalar_types(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "boolean-artifact-schema"
            write_mlir_cublas_ptx_artifact(destination, self.program)
            _rewrite_manifest(
                destination / MANIFEST_FILE,
                lambda value: value.update({"artifact_schema_version": True}),
            )
            with self.assertRaisesRegex(BackendError, "schema version"):
                load_mlir_cublas_ptx_artifact(destination, self.graph)

            destination = root / "boolean-runtime-schema"
            write_mlir_cublas_ptx_artifact(destination, self.program)
            _rewrite_manifest(
                destination / MANIFEST_FILE,
                lambda value: value.update({"runtime_abi_version": True}),
            )
            with self.assertRaisesRegex(BackendError, "does not match"):
                load_mlir_cublas_ptx_artifact(destination, self.graph)

    def test_publication_failure_leaves_no_partial_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "deployment"
            with patch(
                "gac.mlir_cublas_ptx_artifact.os.replace",
                side_effect=OSError("injected rename failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected rename failure"):
                    write_mlir_cublas_ptx_artifact(destination, self.program)
            self.assertFalse(destination.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_refuses_to_replace_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "deployment"
            destination.mkdir()
            with self.assertRaisesRegex(ValueError, "must not already exist"):
                write_mlir_cublas_ptx_artifact(destination, self.program)


if __name__ == "__main__":
    unittest.main()
