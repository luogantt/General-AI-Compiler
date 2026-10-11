"""Tests for the Python 3.9 compatible UniAD FFN bundle protocol."""

from __future__ import annotations

import ast
from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import numpy as np

from gac.uniad.artifact import (
    ArtifactError,
    BUNDLE_FORMAT,
    load_bundle,
    write_bundle,
)
from gac.uniad.contract import ContractError, UniADFFNContract


def small_contract() -> UniADFFNContract:
    return UniADFFNContract(
        layer=2,
        batch=1,
        bev_h=2,
        bev_w=3,
        embed_dim=4,
        hidden_dim=8,
    )


def make_weights(contract: UniADFFNContract) -> dict:
    result = {}
    offset = 1.0
    for name, shape in contract.weight_shapes.items():
        size = int(np.prod(shape))
        result[name] = (
            np.arange(size, dtype=np.float32).reshape(shape) / max(size, 1)
            + offset
        )
        offset += 1.0
    return result


def make_sample(contract: UniADFFNContract) -> dict:
    size = int(np.prod(contract.input_shape))
    query = np.linspace(-1.0, 1.0, size, dtype=np.float32).reshape(
        contract.input_shape
    )
    return {"query": query, "output": np.ascontiguousarray(query * 0.5)}


def read_members(path: Path) -> dict:
    with zipfile.ZipFile(str(path), mode="r") as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}


def write_members(path: Path, members: dict) -> None:
    with zipfile.ZipFile(str(path), mode="w") as archive:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, data)


def update_manifest_record(members: dict, name: str) -> None:
    manifest = json.loads(members["manifest.json"].decode("utf-8"))
    manifest["files"][name] = {
        "sha256": hashlib.sha256(members[name]).hexdigest(),
        "size_bytes": len(members[name]),
    }
    members["manifest.json"] = (
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


class UniADFFNContractTests(unittest.TestCase):
    def test_production_defaults_match_stage2_encoder_ffn(self):
        contract = UniADFFNContract()
        self.assertEqual(contract.stage, "stage2_e2e")
        self.assertEqual(contract.component, "bev_encoder_ffn")
        self.assertEqual(contract.input_shape, (1, 40000, 256))
        self.assertEqual(contract.output_shape, (1, 40000, 256))
        self.assertEqual(contract.hidden_dim, 512)
        self.assertEqual(contract.weight_shapes["fc1_weight"], (256, 512))
        self.assertEqual(contract.weight_shapes["fc2_weight"], (512, 256))
        self.assertEqual(contract.dtype, "float32")

    def test_small_shapes_require_hidden_width_twice_embedding_width(self):
        self.assertEqual(small_contract().input_shape, (1, 6, 4))
        with self.assertRaisesRegex(ContractError, "2 \\* embed_dim"):
            UniADFFNContract(bev_h=2, bev_w=3, embed_dim=4, hidden_dim=7)
        for arguments in (
            {"layer": True},
            {"layer": 6},
            {"batch": 0},
            {"bev_h": -1},
            {"embed_dim": 0, "hidden_dim": 0},
            {"layer_norm_epsilon": float("inf")},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ContractError):
                UniADFFNContract(**arguments)

    def test_contract_json_schema_is_strict(self):
        value = small_contract().to_dict()
        self.assertEqual(UniADFFNContract.from_dict(value), small_contract())
        extra = dict(value)
        extra["unexpected"] = 1
        with self.assertRaisesRegex(ContractError, "extra"):
            UniADFFNContract.from_dict(extra)
        wrong_semantics = dict(value)
        wrong_semantics["dropout"] = "enabled"
        with self.assertRaisesRegex(ContractError, "dropout"):
            UniADFFNContract.from_dict(wrong_semantics)
        for invalid_version in (True, 1.0, "1"):
            invalid = dict(value)
            invalid["schema_version"] = invalid_version
            with self.subTest(schema_version=invalid_version):
                with self.assertRaisesRegex(ContractError, "schema_version"):
                    UniADFFNContract.from_dict(invalid)

    def test_protocol_modules_parse_as_python_39(self):
        package = Path(__file__).parents[1] / "src" / "gac" / "uniad"
        for name in ("contract.py", "artifact.py", "checkpoint.py", "__init__.py"):
            with self.subTest(name=name):
                ast.parse(
                    (package / name).read_text(encoding="utf-8"),
                    filename=name,
                    feature_version=(3, 9),
                )

    def test_official_environment_entrypoints_parse_as_python_39(self):
        root = Path(__file__).parents[1]
        paths = (
            root / "src" / "gac" / "tools" / "uniad_export_ffn.py",
            root / "src" / "gac" / "tools" / "uniad_benchmark_e2e.py",
            root / "tools" / "uniad_export_ffn.py",
            root / "tools" / "uniad_benchmark_e2e.py",
        )
        for path in paths:
            with self.subTest(path=path.name):
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                    feature_version=(3, 9),
                )


class UniADFFNArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "layer2.uniadffn"
        self.contract = small_contract()
        self.weights = make_weights(self.contract)
        self.sample = make_sample(self.contract)

    def test_round_trip_with_sample_and_manifest_hashes(self):
        written = write_bundle(
            self.path, self.contract, self.weights, sample=self.sample
        )
        loaded = load_bundle(self.path)

        self.assertEqual(written.contract, self.contract)
        self.assertEqual(loaded.contract, self.contract)
        self.assertEqual(loaded.manifest["format"], BUNDLE_FORMAT)
        self.assertEqual(
            set(read_members(self.path)),
            {"contract.json", "weights.npz", "sample.npz", "manifest.json"},
        )
        self.assertEqual(
            set(loaded.manifest["files"]),
            {"contract.json", "weights.npz", "sample.npz"},
        )
        for name in self.weights:
            np.testing.assert_array_equal(loaded.weights[name], self.weights[name])
            self.assertEqual(loaded.weights[name].dtype, np.float32)
            self.assertTrue(loaded.weights[name].flags.c_contiguous)
            self.assertFalse(loaded.weights[name].flags.writeable)
        for name in self.sample:
            np.testing.assert_array_equal(loaded.sample[name], self.sample[name])

        members = read_members(self.path)
        for name, record in loaded.manifest["files"].items():
            self.assertEqual(record["size_bytes"], len(members[name]))
            self.assertEqual(record["sha256"], hashlib.sha256(members[name]).hexdigest())

    def test_round_trip_without_optional_sample(self):
        write_bundle(self.path, self.contract, self.weights)
        loaded = load_bundle(self.path)
        self.assertIsNone(loaded.sample)
        self.assertEqual(
            set(read_members(self.path)),
            {"contract.json", "weights.npz", "manifest.json"},
        )

    def test_production_weight_shapes_round_trip(self):
        contract = UniADFFNContract()
        weights = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in contract.weight_shapes.items()
        }
        write_bundle(self.path, contract, weights)
        loaded = load_bundle(self.path)
        self.assertEqual(loaded.contract.input_shape, (1, 40000, 256))
        self.assertEqual(loaded.weights["fc1_weight"].shape, (256, 512))
        self.assertEqual(loaded.weights["fc2_weight"].shape, (512, 256))

    def test_identical_inputs_produce_identical_bundle_bytes(self):
        second = Path(self.temporary.name) / "second.uniadffn"
        write_bundle(self.path, self.contract, self.weights, self.sample)
        write_bundle(second, self.contract, self.weights, self.sample)
        self.assertEqual(self.path.read_bytes(), second.read_bytes())

    def test_writer_rejects_weight_key_shape_dtype_finite_and_layout_errors(self):
        cases = []
        missing = dict(self.weights)
        missing.pop("norm_bias")
        cases.append(missing)
        extra = dict(self.weights)
        extra["extra"] = np.zeros((1,), dtype=np.float32)
        cases.append(extra)
        wrong_shape = dict(self.weights)
        wrong_shape["fc1_bias"] = np.zeros((7,), dtype=np.float32)
        cases.append(wrong_shape)
        wrong_dtype = dict(self.weights)
        wrong_dtype["norm_weight"] = np.zeros((4,), dtype=np.float64)
        cases.append(wrong_dtype)
        nonfinite = dict(self.weights)
        nonfinite["fc2_weight"] = nonfinite["fc2_weight"].copy()
        nonfinite["fc2_weight"][0, 0] = np.nan
        cases.append(nonfinite)
        noncontiguous = dict(self.weights)
        noncontiguous["fc1_weight"] = np.zeros((8, 4), dtype=np.float32).T
        cases.append(noncontiguous)

        for index, invalid in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ArtifactError):
                write_bundle(self.path, self.contract, invalid)

    def test_writer_rejects_invalid_sample(self):
        for invalid in (
            {"query": self.sample["query"]},
            {
                "query": self.sample["query"],
                "output": np.zeros((1, 5, 4), dtype=np.float32),
            },
            {
                "query": self.sample["query"].astype(np.float64),
                "output": self.sample["output"],
            },
        ):
            with self.subTest(keys=sorted(invalid)), self.assertRaises(ArtifactError):
                write_bundle(self.path, self.contract, self.weights, invalid)

    def test_failed_write_does_not_replace_existing_file(self):
        self.path.write_bytes(b"existing")
        invalid = dict(self.weights)
        invalid["norm_bias"] = np.array([np.inf] * 4, dtype=np.float32)
        with self.assertRaises(ArtifactError):
            write_bundle(self.path, self.contract, invalid)
        self.assertEqual(self.path.read_bytes(), b"existing")

    def test_replace_failure_preserves_destination_and_cleans_temporary_file(self):
        self.path.write_bytes(b"existing")
        with patch(
            "gac.uniad.artifact.os.replace",
            side_effect=OSError("replace failed"),
        ):
            with self.assertRaisesRegex(ArtifactError, "replace failed"):
                write_bundle(self.path, self.contract, self.weights)
        self.assertEqual(self.path.read_bytes(), b"existing")
        self.assertEqual(
            [entry for entry in self.path.parent.iterdir() if entry != self.path], []
        )

    def test_payload_tampering_is_rejected_before_numpy_load(self):
        write_bundle(self.path, self.contract, self.weights)
        members = read_members(self.path)
        corrupted = bytearray(members["weights.npz"])
        corrupted[-1] ^= 0x1
        members["weights.npz"] = bytes(corrupted)
        write_members(self.path, members)
        with self.assertRaisesRegex(ArtifactError, "SHA256"):
            load_bundle(self.path)

    def test_semantically_invalid_contract_is_rejected_even_with_valid_hash(self):
        write_bundle(self.path, self.contract, self.weights)
        members = read_members(self.path)
        contract = json.loads(members["contract.json"].decode("utf-8"))
        contract["hidden_dim"] = 7
        members["contract.json"] = (
            json.dumps(contract, separators=(",", ":"), sort_keys=True).encode("utf-8")
            + b"\n"
        )
        update_manifest_record(members, "contract.json")
        write_members(self.path, members)
        with self.assertRaisesRegex(ArtifactError, "hidden_dim"):
            load_bundle(self.path)

    def test_manifest_schema_version_requires_an_integer(self):
        write_bundle(self.path, self.contract, self.weights)
        for invalid_version in (True, 1.0, "1"):
            with self.subTest(schema_version=invalid_version):
                members = read_members(self.path)
                manifest = json.loads(members["manifest.json"].decode("utf-8"))
                manifest["schema_version"] = invalid_version
                members["manifest.json"] = (
                    json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode(
                        "utf-8"
                    )
                    + b"\n"
                )
                candidate = self.path.with_name(
                    "manifest-{}.uniadffn".format(repr(invalid_version))
                )
                write_members(candidate, members)
                with self.assertRaisesRegex(ArtifactError, "schema version"):
                    load_bundle(candidate)

    def test_object_dtype_npz_is_rejected_with_allow_pickle_false(self):
        write_bundle(self.path, self.contract, self.weights)
        members = read_members(self.path)
        malicious = dict(self.weights)
        malicious["fc1_bias"] = np.array([object()] * 8, dtype=object)
        output = BytesIO()
        np.savez(output, **malicious)
        members["weights.npz"] = output.getvalue()
        update_manifest_record(members, "weights.npz")
        write_members(self.path, members)
        with self.assertRaisesRegex(ArtifactError, "decode weights.npz"):
            load_bundle(self.path)

    def test_extra_and_duplicate_outer_members_are_rejected(self):
        write_bundle(self.path, self.contract, self.weights)
        members = read_members(self.path)
        members["extra.txt"] = b"no"
        write_members(self.path, members)
        with self.assertRaisesRegex(ArtifactError, "member set"):
            load_bundle(self.path)

        write_bundle(self.path, self.contract, self.weights)
        members = read_members(self.path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(str(self.path), mode="a") as archive:
                archive.writestr("contract.json", members["contract.json"])
        with self.assertRaisesRegex(ArtifactError, "duplicate"):
            load_bundle(self.path)


if __name__ == "__main__":
    unittest.main()
