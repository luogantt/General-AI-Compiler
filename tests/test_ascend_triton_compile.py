from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from collections import namedtuple
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from gac.core.errors import BackendError, DeviceUnavailableError, ValidationError
from gac.targets.ascend import triton_compile


_TTIR = b'''module {
  "tt.func"() ({}) {sym_name = "gac_op"} : () -> ()
}
'''


def _manifest(*filenames: str) -> dict:
    rows = []
    for index, filename in enumerate(filenames):
        rows.append(
            {
                "kernel": f"gac_op_{index}",
                "lowering": "linalg-to-triton",
                "compute_template": {
                    "id": "triton.standard.pointwise-fp32",
                    "selected": True,
                },
                "triton": {
                    "kind": "pointwise",
                    "ttir_file": filename,
                    "artifact_format": "standard-ttir",
                    "ttir_signature": "ptr<input:f32>, ptr<output:f32>",
                },
            }
        )
    return {
        "schema_version": 1,
        "target": "ascend",
        "artifact_format": "standard-ttir",
        "execution": "artifact-only",
        "runtime_available": False,
        "triton_dispatches": len(rows),
        "dispatches": rows,
    }


class AscendManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = self.root / "ttir"
        self.sources.mkdir()
        self.manifest = self.root / "plan.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_fixture(self, *filenames: str) -> None:
        for filename in set(filenames):
            (self.sources / filename).write_bytes(_TTIR)
        self.manifest.write_text(
            json.dumps(_manifest(*filenames)), encoding="utf-8"
        )

    def test_valid_manifest_snapshots_all_sources(self) -> None:
        self.write_fixture("gac_op_0.ttir", "gac_op_1.ttir")
        validated = triton_compile.validate_ascend_manifest(
            self.manifest, self.sources
        )
        self.assertEqual(len(validated.dispatches), 2)
        self.assertEqual(validated.dispatches[0].source_bytes, _TTIR)
        self.assertEqual(len(validated.dispatches[0].source_sha256), 64)

    def test_root_contract_is_fail_closed(self) -> None:
        self.write_fixture("gac_op_0.ttir")
        payload = _manifest("gac_op_0.ttir")
        for field, bad_value in (
            ("schema_version", 2),
            ("target", "vortex"),
            ("artifact_format", "npubin"),
            ("execution", "runtime"),
            ("runtime_available", True),
        ):
            with self.subTest(field=field):
                changed = dict(payload)
                changed[field] = bad_value
                self.manifest.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValidationError):
                    triton_compile.validate_ascend_manifest(
                        self.manifest, self.sources
                    )

    def test_every_dispatch_must_be_standard_triton(self) -> None:
        self.write_fixture("gac_op_0.ttir")
        cases = []
        missing_triton = _manifest("gac_op_0.ttir")
        del missing_triton["dispatches"][0]["triton"]
        cases.append(missing_triton)
        native = _manifest("gac_op_0.ttir")
        native["dispatches"][0]["lowering"] = "gac-specialized"
        cases.append(native)
        vortex = _manifest("gac_op_0.ttir")
        vortex["dispatches"][0]["triton"]["num_warps"] = 4
        cases.append(vortex)
        unselected = _manifest("gac_op_0.ttir")
        unselected["dispatches"][0]["compute_template"]["selected"] = False
        cases.append(unselected)
        wrong_count = _manifest("gac_op_0.ttir")
        wrong_count["triton_dispatches"] = 0
        cases.append(wrong_count)
        for payload in cases:
            with self.subTest(payload=payload):
                self.manifest.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValidationError):
                    triton_compile.validate_ascend_manifest(
                        self.manifest, self.sources
                    )

    def test_filenames_are_portable_unique_and_confined(self) -> None:
        (self.sources / "gac_op_0.ttir").write_bytes(_TTIR)
        unsafe_names = ("../escape.ttir", "nested/file.ttir", "bad name.ttir")
        for filename in unsafe_names:
            with self.subTest(filename=filename):
                self.manifest.write_text(
                    json.dumps(_manifest(filename)), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValidationError, "unsafe|directory"):
                    triton_compile.validate_ascend_manifest(
                        self.manifest, self.sources
                    )

        duplicate = _manifest("gac_op_0.ttir", "GAC_OP_0.ttir")
        duplicate["dispatches"][1]["kernel"] = "another_kernel"
        self.manifest.write_text(json.dumps(duplicate), encoding="utf-8")
        with self.assertRaisesRegex(ValidationError, "duplicate.*filename"):
            triton_compile.validate_ascend_manifest(self.manifest, self.sources)

    def test_all_sources_are_preflighted_before_vendor_import(self) -> None:
        (self.sources / "gac_op_0.ttir").write_bytes(_TTIR)
        self.manifest.write_text(
            json.dumps(_manifest("gac_op_0.ttir", "missing.ttir")),
            encoding="utf-8",
        )
        with mock.patch.object(triton_compile, "_load_vendor_stack") as load:
            with self.assertRaisesRegex(ValidationError, "does not exist"):
                triton_compile.compile_ascend_ttir_manifest(
                    self.manifest, self.sources, self.root / "output"
                )
        load.assert_not_called()


class AscendCompilationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.sources = self.root / "ttir"
        self.sources.mkdir()
        self.manifest = self.root / "plan.json"
        for index in range(2):
            (self.sources / f"gac_op_{index}.ttir").write_bytes(_TTIR)
        self.manifest.write_text(
            json.dumps(_manifest("gac_op_0.ttir", "gac_op_1.ttir")),
            encoding="utf-8",
        )
        target = SimpleNamespace(backend="npu", arch="Ascend910B2", warp_size=32)
        self.vendor = SimpleNamespace(
            torch_version="2.7.1",
            torch_npu_version="2.7.1.post4",
            triton_version="3.2.0",
            triton_ascend_version="3.2.1",
            driver=SimpleNamespace(
                active=SimpleNamespace(get_current_target=lambda: target)
            ),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def fake_compile(dispatch, output, root, vendor, target):
        normalized = output / "normalized.ttir"
        adapter = output / "kernel.ttadapter.mlir"
        binary = output / "kernel.npubin"
        metadata = output / "metadata.json"
        normalized.write_bytes(dispatch.source_bytes)
        adapter.write_text("module {}\n", encoding="utf-8")
        binary.write_bytes(b"NPUBIN")
        metadata.write_text('{"kernel_name":"mock"}\n', encoding="utf-8")
        return {
            "dispatch_index": dispatch.index,
            "manifest_kernel": dispatch.kernel,
            "launcher_signature": {"0": "*fp32"},
        }

    def test_success_publishes_complete_directory_and_restores_environment(self) -> None:
        destination = self.root / "compiled"
        old_cache = os.environ.get("TRITON_CACHE_DIR")
        old_always = os.environ.get("TRITON_ALWAYS_COMPILE")
        with mock.patch.object(
            triton_compile, "_load_vendor_stack", return_value=self.vendor
        ), mock.patch.object(
            triton_compile, "_compile_vendor_dispatch", side_effect=self.fake_compile
        ) as compile_dispatch:
            report = triton_compile.compile_ascend_ttir_manifest(
                self.manifest,
                self.sources,
                destination,
                cache_dir=self.root / "cache",
                always_compile=True,
            )
        self.assertEqual(compile_dispatch.call_count, 2)
        self.assertEqual(report["dispatch_count"], 2)
        self.assertEqual(report["versions"]["triton"], "3.2.0")
        self.assertEqual(report["versions"]["triton_ascend"], "3.2.1")
        self.assertTrue((destination / "compile-manifest.json").is_file())
        self.assertTrue(
            (destination / "dispatches" / "0001-gac_op_1" / "kernel.npubin").is_file()
        )
        self.assertEqual(os.environ.get("TRITON_CACHE_DIR"), old_cache)
        self.assertEqual(os.environ.get("TRITON_ALWAYS_COMPILE"), old_always)
        self.assertFalse(any(self.root.glob(".compiled-*")))

    def test_dispatch_failure_leaves_no_output_or_staging_tree(self) -> None:
        destination = self.root / "compiled"
        calls = 0

        def fail_second(dispatch, output, root, vendor, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("vendor compiler failed")
            return self.fake_compile(dispatch, output, root, vendor, target)

        with mock.patch.object(
            triton_compile, "_load_vendor_stack", return_value=self.vendor
        ), mock.patch.object(
            triton_compile, "_compile_vendor_dispatch", side_effect=fail_second
        ):
            with self.assertRaisesRegex(BackendError, "dispatch 1"):
                triton_compile.compile_ascend_ttir_manifest(
                    self.manifest, self.sources, destination
                )
        self.assertFalse(destination.exists())
        self.assertFalse(any(self.root.glob(".compiled-*")))

    def test_existing_destination_is_never_overwritten(self) -> None:
        destination = self.root / "compiled"
        destination.mkdir()
        marker = destination / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with mock.patch.object(triton_compile, "_load_vendor_stack") as load:
            with self.assertRaisesRegex(ValidationError, "refusing to overwrite"):
                triton_compile.compile_ascend_ttir_manifest(
                    self.manifest, self.sources, destination
                )
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        load.assert_not_called()

    def test_dangling_output_symlink_is_rejected(self) -> None:
        destination = self.root / "compiled"
        try:
            destination.symlink_to(self.root / "missing-target", target_is_directory=True)
        except (NotImplementedError, OSError):
            self.skipTest("symbolic links are unavailable on this host")
        with mock.patch.object(triton_compile, "_load_vendor_stack") as load:
            with self.assertRaisesRegex(ValidationError, "symbolic link"):
                triton_compile.compile_ascend_ttir_manifest(
                    self.manifest, self.sources, destination
                )
        load.assert_not_called()

    def test_cache_must_be_outside_atomic_output_tree(self) -> None:
        destination = self.root / "compiled"
        for cache in (destination, destination / ".triton-cache"):
            with self.subTest(cache=cache), mock.patch.object(
                triton_compile, "_load_vendor_stack"
            ) as load:
                with self.assertRaisesRegex(ValidationError, "cache directory.*outside"):
                    triton_compile.compile_ascend_ttir_manifest(
                        self.manifest,
                        self.sources,
                        destination,
                        cache_dir=cache,
                    )
                load.assert_not_called()
                self.assertFalse(destination.exists())

    def test_inherited_cache_must_be_outside_atomic_output_tree(self) -> None:
        destination = self.root / "compiled"
        inherited_cache = destination / ".triton-cache"
        with mock.patch.dict(
            os.environ, {"TRITON_CACHE_DIR": str(inherited_cache)}
        ), mock.patch.object(triton_compile, "_load_vendor_stack") as load:
            with self.assertRaisesRegex(ValidationError, "cache directory.*outside"):
                triton_compile.compile_ascend_ttir_manifest(
                    self.manifest, self.sources, destination
                )
        load.assert_not_called()
        self.assertFalse(destination.exists())

    def test_active_target_must_be_supported_npu(self) -> None:
        for backend, arch in (("cuda", "sm_80"), ("npu", "Ascend910")):
            with self.subTest(backend=backend, arch=arch):
                target = SimpleNamespace(backend=backend, arch=arch, warp_size=32)
                vendor = SimpleNamespace(
                    driver=SimpleNamespace(
                        active=SimpleNamespace(get_current_target=lambda: target)
                    )
                )
                with self.assertRaises(DeviceUnavailableError):
                    triton_compile._active_target(vendor)

    def test_active_target_accepts_official_product_identifiers(self) -> None:
        for arch in ("Ascend910B2", "Ascend910_9391", "Ascend950PR_950z"):
            with self.subTest(arch=arch):
                target = SimpleNamespace(backend="npu", arch=arch, warp_size=0)
                vendor = SimpleNamespace(
                    driver=SimpleNamespace(
                        active=SimpleNamespace(get_current_target=lambda: target)
                    )
                )
                _, report = triton_compile._active_target(vendor)
                self.assertEqual(report["arch"], arch)


class AscendVendorBoundaryTests(unittest.TestCase):
    def test_launcher_signature_canonicalizes_triton_32_pointer_bug(self) -> None:
        self.assertEqual(triton_compile.canonical_launcher_type("*f32>"), "*fp32")
        self.assertEqual(triton_compile.canonical_launcher_type("*f16> loc(x)"), "*fp16")
        self.assertEqual(triton_compile.canonical_launcher_type("!tt.ptr<f32>"), "*fp32")
        self.assertEqual(triton_compile.canonical_launcher_type("f32"), "fp32")
        self.assertEqual(triton_compile.canonical_launcher_type("i32"), "i32")

    def test_optional_dependency_error_is_actionable(self) -> None:
        for error in (ImportError("no torch_npu"), RuntimeError("CANN version skew")):
            with self.subTest(error=type(error).__name__), mock.patch.object(
                importlib, "import_module", side_effect=error
            ):
                with self.assertRaisesRegex(
                    DeviceUnavailableError, "torch_npu.*Triton-Ascend"
                ):
                    triton_compile._load_vendor_stack()

    def test_namedtuple_metadata_keeps_field_names_and_target(self) -> None:
        Target = namedtuple("Target", "backend arch warp_size")
        Metadata = namedtuple("Metadata", "kernel_name target num_warps")
        metadata = Metadata("gac_op", Target("npu", "Ascend910B2", 32), 4)
        self.assertEqual(
            triton_compile._metadata_object(metadata),
            {
                "kernel_name": "gac_op",
                "target": {
                    "backend": "npu",
                    "arch": "Ascend910B2",
                    "warp_size": 32,
                },
                "num_warps": 4,
            },
        )

    def test_vendor_dispatch_rejects_empty_signature_or_wrong_kernel_name(self) -> None:
        dispatch = triton_compile.AscendTTIRDispatch(
            index=0,
            kernel="gac_op_0",
            ttir_file="gac_op_0.ttir",
            source_path=Path("gac_op_0.ttir"),
            source_bytes=_TTIR,
            source_sha256="0" * 64,
        )
        target = SimpleNamespace(backend="npu", arch="Ascend910B2", warp_size=0)
        for signature, kernel_name, message in (
            ({}, "gac_op_0", "empty launcher signature"),
            ({0: "*f32>"}, "different_kernel", "does not match"),
        ):
            with self.subTest(signature=signature, kernel_name=kernel_name):
                kernel = SimpleNamespace(
                    asm={"npubin": b"binary", "ttadapter": "module {}\n"},
                    src=SimpleNamespace(signature=signature),
                    metadata=SimpleNamespace(kernel_name=kernel_name),
                )
                vendor = SimpleNamespace(compile=lambda path, result=kernel: result)
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    output = root / "dispatch"
                    output.mkdir()
                    with mock.patch.object(
                        triton_compile,
                        "_normalize_ttir",
                        side_effect=lambda source, destination, vendor, target: (
                            destination.write_bytes(_TTIR) or _TTIR
                        ),
                    ):
                        with self.assertRaisesRegex(BackendError, message):
                            triton_compile._compile_vendor_dispatch(
                                dispatch, output, root, vendor, target
                            )

    def test_module_import_does_not_load_vendor_packages(self) -> None:
        source = Path(triton_compile.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import torch\n", source)
        self.assertNotIn("import torch_npu\n", source)
        self.assertNotIn("import triton\n", source)


if __name__ == "__main__":
    unittest.main()
