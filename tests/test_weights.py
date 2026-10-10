from __future__ import annotations

from contextlib import redirect_stderr
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import warnings
import zipfile

import numpy as np

from gac.errors import ValidationError
from gac.tools import export_bevformer_checkpoint as exporter
from gac.weights import load_weights, write_weights


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIGEST = "42" * 32


def read_members(path: Path) -> list[tuple[str, bytes]]:
    with zipfile.ZipFile(path, "r") as archive:
        return [(info.filename, archive.read(info)) for info in archive.infolist()]


def rewrite(path: Path, destination: Path, transform, *, compression=zipfile.ZIP_STORED) -> None:
    members = transform(read_members(path))
    with zipfile.ZipFile(destination, "w", compression=compression) as archive:
        for name, data in members:
            archive.writestr(name, data)


def rewrite_manifest(members, mutation):
    result = []
    for name, data in members:
        if name == "manifest.json":
            manifest = json.loads(data)
            mutation(manifest)
            data = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        result.append((name, data))
    return result


class WeightsFormatTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.values = {
            "img_neck.bias": np.asarray([1.5, -2.0], dtype=np.float32),
            "img_backbone.layer.weight": np.arange(12, dtype=np.float16).reshape(3, 4),
            "img_backbone.flag": np.asarray(True),
        }

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name="model.gacweights") -> Path:
        return write_weights(
            self.root / name,
            self.values,
            source_checkpoint_sha256=SOURCE_DIGEST.upper(),
        )

    def test_deterministic_layout_round_trip_and_read_only_contract(self):
        first, second = self.write("first.gacweights"), self.write("second.gacweights")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            self.assertEqual(
                archive.namelist(),
                [
                    "manifest.json",
                    "weights/000000.npy",
                    "weights/000001.npy",
                    "weights/000002.npy",
                ],
            )
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["format"], "gacweights")
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["source_checkpoint_sha256"], SOURCE_DIGEST)
            for record in manifest["weights"].values():
                self.assertEqual(set(record), {"path", "shape", "dtype", "size", "sha256"})
                self.assertEqual(len(record["sha256"]), 64)

        loaded = load_weights(first)
        self.assertEqual(loaded.source_checkpoint_sha256, SOURCE_DIGEST)
        self.assertEqual(set(loaded), set(self.values))
        for name, expected in self.values.items():
            np.testing.assert_array_equal(loaded[name], expected)
            self.assertFalse(loaded[name].flags.writeable)
            with self.assertRaises(ValueError):
                loaded[name].flat[0] = 0
            with self.assertRaises(ValueError):
                loaded[name].setflags(write=True)
        with self.assertRaises(TypeError):
            loaded.weights["new"] = np.zeros(1)  # type: ignore[index]

    def test_writer_rejects_bad_digest_names_types_and_values(self):
        with self.assertRaisesRegex(ValidationError, "64-character"):
            write_weights(self.root / "x", self.values, source_checkpoint_sha256="bad")
        with self.assertRaisesRegex(ValidationError, "at least one"):
            write_weights(self.root / "x", {}, source_checkpoint_sha256=SOURCE_DIGEST)
        with self.assertRaisesRegex(ValidationError, "names"):
            write_weights(self.root / "x", {"": np.zeros(1)}, source_checkpoint_sha256=SOURCE_DIGEST)
        with self.assertRaisesRegex(ValidationError, "NumPy ndarray"):
            write_weights(self.root / "x", {"x": [1, 2]}, source_checkpoint_sha256=SOURCE_DIGEST)  # type: ignore[dict-item]
        with self.assertRaisesRegex(ValidationError, "object dtype"):
            write_weights(
                self.root / "x",
                {"x": np.asarray([object()], dtype=object)},
                source_checkpoint_sha256=SOURCE_DIGEST,
            )
        with self.assertRaisesRegex(ValidationError, "non-finite"):
            write_weights(
                self.root / "x",
                {"x": np.asarray([np.nan], dtype=np.float32)},
                source_checkpoint_sha256=SOURCE_DIGEST,
            )

    def test_rejects_corruption_schema_and_provenance(self):
        good = self.write()
        corrupt = self.root / "corrupt.gacweights"
        rewrite(
            good,
            corrupt,
            lambda members: [
                (name, data + b"tampered" if name == "weights/000000.npy" else data)
                for name, data in members
            ],
        )
        with self.assertRaisesRegex(ValidationError, "wrong member size|SHA256"):
            load_weights(corrupt)

        schema = self.root / "schema.gacweights"
        rewrite(good, schema, lambda members: rewrite_manifest(members, lambda m: m.update(schema_version=9)))
        with self.assertRaisesRegex(ValidationError, "unsupported weights schema"):
            load_weights(schema)

        digest = self.root / "digest.gacweights"
        rewrite(
            good,
            digest,
            lambda members: rewrite_manifest(
                members, lambda m: m.update(source_checkpoint_sha256="invalid")
            ),
        )
        with self.assertRaisesRegex(ValidationError, "64-character"):
            load_weights(digest)

    def test_rejects_path_traversal_duplicates_extras_and_compression(self):
        good = self.write()
        traversal = self.root / "traversal.gacweights"
        rewrite(good, traversal, lambda members: members + [("../escape.npy", b"bad")])
        with self.assertRaisesRegex(ValidationError, "unsafe member path"):
            load_weights(traversal)

        duplicate = self.root / "duplicate.gacweights"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rewrite(good, duplicate, lambda members: members + [("weights/000000.npy", b"bad")])
        with self.assertRaisesRegex(ValidationError, "duplicate member"):
            load_weights(duplicate)

        extra = self.root / "extra.gacweights"
        rewrite(good, extra, lambda members: members + [("extra.bin", b"bad")])
        with self.assertRaisesRegex(ValidationError, "differs from manifest"):
            load_weights(extra)

        compressed = self.root / "compressed.gacweights"
        rewrite(good, compressed, lambda members: members, compression=zipfile.ZIP_DEFLATED)
        with self.assertRaisesRegex(ValidationError, "unsupported compression"):
            load_weights(compressed)

    def test_rejects_resource_limits_and_unsafe_npy_even_with_updated_hash(self):
        good = self.write()
        with self.assertRaisesRegex(ValidationError, "size limit"):
            load_weights(good, max_uncompressed_bytes=16)
        with self.assertRaisesRegex(ValidationError, "too many members"):
            load_weights(good, max_weight_count=2)

        stream = io.BytesIO()
        np.save(stream, np.asarray([object()], dtype=object), allow_pickle=True)
        object_npy = stream.getvalue()
        unsafe = self.root / "unsafe.gacweights"

        def replace_with_object(members):
            updated = []
            for name, data in members:
                if name == "weights/000000.npy":
                    data = object_npy
                updated.append((name, data))

            def update(manifest):
                record = next(
                    item for item in manifest["weights"].values() if item["path"] == "weights/000000.npy"
                )
                record.update(
                    shape=[1],
                    dtype="|O",
                    size=len(object_npy),
                    sha256=sha256(object_npy).hexdigest(),
                )

            return rewrite_manifest(updated, update)

        rewrite(good, unsafe, replace_with_object)
        with self.assertRaisesRegex(ValidationError, "unsafe|unsupported"):
            load_weights(unsafe)


class FakeDevice:
    type = "cpu"


class FakeTensor:
    is_sparse = False
    is_quantized = False
    layout = "strided"
    device = FakeDevice()

    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self.value


class FakeTorch:
    strided = "strided"

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)

    def load(self, path, map_location=None, weights_only=False):
        if map_location != "cpu":
            raise AssertionError("checkpoint must be loaded on CPU")
        return self.checkpoint


class CheckpointExporterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "checkpoint.pth"
        self.checkpoint.write_bytes(b"trusted-checkpoint-fixture")
        self.digest = sha256(self.checkpoint.read_bytes()).hexdigest()

    def tearDown(self):
        self.temporary.cleanup()

    def test_module_import_and_help_do_not_import_torch(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        code = (
            "import sys; import gac.tools.export_bevformer_checkpoint; "
            "assert 'torch' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            [sys.executable, "-m", "gac.tools.export_bevformer_checkpoint", "--help"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--trusted-checkpoint", result.stdout)

    def test_cli_requires_explicit_trusted_checkpoint(self):
        parser = exporter.build_parser()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--output", str(self.root / "x.gacweights")])

    def test_expected_digest_is_checked_before_torch_import(self):
        with mock.patch.object(exporter, "_load_torch") as load_torch:
            with self.assertRaisesRegex(ValidationError, "SHA256 mismatch"):
                exporter.export_bevformer_checkpoint(
                    self.checkpoint,
                    self.root / "x.gacweights",
                    expected_sha256="00" * 32,
                )
            load_torch.assert_not_called()

    def test_missing_torch_has_clear_error_after_hashing(self):
        with mock.patch.object(exporter.importlib, "import_module", side_effect=ModuleNotFoundError("torch")):
            with self.assertRaisesRegex(ValidationError, "official BEVFormer environment"):
                exporter.export_bevformer_checkpoint(
                    self.checkpoint,
                    self.root / "x.gacweights",
                    expected_sha256=self.digest,
                )

    def test_default_filter_module_prefix_and_state_dict_export(self):
        fake = FakeTorch(
            {
                "state_dict": {
                    "module.img_backbone.conv.weight": FakeTensor(np.arange(4, dtype=np.float32)),
                    "module.img_neck.bias": FakeTensor(np.asarray([2], dtype=np.int64)),
                    "module.pts_bbox_head.ignore": FakeTensor(np.asarray([3], dtype=np.float32)),
                }
            }
        )
        output = self.root / "selected.gacweights"
        with mock.patch.object(exporter, "_load_torch", return_value=fake):
            report = exporter.export_bevformer_checkpoint(
                self.checkpoint,
                output,
                expected_sha256=self.digest.upper(),
            )
        self.assertEqual(report["weight_count"], 3)
        loaded = load_weights(output)
        self.assertEqual(
            set(loaded),
            {
                "img_backbone.conv.weight",
                "img_neck.bias",
                "pts_bbox_head.ignore",
            },
        )
        self.assertEqual(loaded.source_checkpoint_sha256, self.digest)

    def test_custom_prefix_and_duplicate_normalized_key_rejection(self):
        fake = FakeTorch(
            {
                "module.custom.weight": FakeTensor([1]),
                "other": FakeTensor([2]),
            }
        )
        with mock.patch.object(exporter, "_load_torch", return_value=fake):
            exporter.export_bevformer_checkpoint(
                self.checkpoint,
                self.root / "custom.gacweights",
                prefixes=["custom."],
            )
        self.assertEqual(set(load_weights(self.root / "custom.gacweights")), {"custom.weight"})

        duplicate = FakeTorch(
            {
                "state_dict": {
                    "module.img_backbone.weight": FakeTensor([1]),
                    "img_backbone.weight": FakeTensor([2]),
                }
            }
        )
        with mock.patch.object(exporter, "_load_torch", return_value=duplicate):
            with self.assertRaisesRegex(ValidationError, "duplicate key"):
                exporter.export_bevformer_checkpoint(
                    self.checkpoint,
                    self.root / "duplicate.gacweights",
                )

    def test_selected_nonfinite_sparse_quantized_and_non_tensor_are_rejected(self):
        cases = {
            "nonfinite": FakeTensor([np.inf]),
            "non_tensor": object(),
        }
        sparse = FakeTensor([1])
        sparse.is_sparse = True
        quantized = FakeTensor([1])
        quantized.is_quantized = True
        cases.update(sparse=sparse, quantized=quantized)
        for label, value in cases.items():
            fake = FakeTorch({"img_backbone.bad": value})
            with self.subTest(label=label), mock.patch.object(exporter, "_load_torch", return_value=fake):
                with self.assertRaises(ValidationError):
                    exporter.export_bevformer_checkpoint(
                        self.checkpoint,
                        self.root / f"{label}.gacweights",
                    )


if __name__ == "__main__":
    unittest.main()
