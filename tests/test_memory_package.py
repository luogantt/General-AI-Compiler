from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

import numpy as np

from gac.backends.cuda import compile_cuda
from gac.errors import ValidationError
from gac.ir import Graph, Node, TensorSpec
from gac.memory import MemoryPlan, plan_memory
from gac.package import (
    load_package,
    rebuild_cuda_program,
    write_model_package,
    write_package,
)


def planning_graph() -> Graph:
    vector = TensorSpec((4,), "float32")
    return Graph(
        "memory-example",
        {"x": vector},
        ("y",),
        [
            Node("make-a", "Identity", ("x",), ("a",)),
            Node("make-b", "Add", ("a", "weight"), ("b",)),
            Node("make-d", "Identity", ("b",), ("d",)),
            Node("make-y", "Identity", ("d",), ("y",)),
        ],
        constants={"weight": np.arange(4, dtype=np.float32)},
        value_specs={name: vector for name in ("a", "b", "d", "y")},
    )


def reshape_view_graph() -> Graph:
    source = TensorSpec((2, 3), "float32")
    reshaped = TensorSpec((3, 2), "float32")
    return Graph(
        "reshape-view",
        {"x": source},
        ("y",),
        [
            Node("materialize", "Identity", ("x",), ("materialized",)),
            Node("reshape", "Reshape", ("materialized",), ("view",), {"shape": [3, 2]}),
            Node("consume", "Identity", ("view",), ("y",)),
        ],
        value_specs={"materialized": source, "view": reshaped, "y": reshaped},
    )


def rewrite_zip(source: Path, destination: Path, transform) -> None:
    with zipfile.ZipFile(source, "r") as archive:
        members = [(info.filename, archive.read(info)) for info in archive.infolist()]
    members = transform(members)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in members:
            archive.writestr(name, data)


class StaticMemoryTests(unittest.TestCase):
    def test_first_fit_liveness_and_storage_contract(self):
        plan = plan_memory(planning_graph(), alignment=256)
        self.assertEqual(plan.allocation("x").storage, "external-input")
        self.assertEqual(plan.allocation("weight").storage, "resident-constant")
        self.assertEqual(plan.allocation("y").storage, "external-output")
        self.assertTrue(plan.allocation("x").external)
        self.assertTrue(plan.allocation("weight").resident)
        self.assertTrue(plan.allocation("y").is_output)

        self.assertEqual(plan.allocation("a").offset, 0)
        self.assertEqual(plan.allocation("b").offset, 256)
        self.assertEqual(plan.allocation("d").offset, 0)
        self.assertEqual(plan.allocation("a").defined_at, 0)
        self.assertEqual(plan.allocation("a").last_use, 1)
        self.assertEqual(plan.allocation("d").defined_at, 2)
        self.assertEqual(plan.arena_size, 272)
        self.assertEqual(plan.peak_live_bytes, 32)
        self.assertEqual(plan.resident_bytes, 16)
        self.assertEqual(plan.external_input_bytes, 16)
        self.assertEqual(plan.external_output_bytes, 16)
        self.assertEqual(plan.buffer_offsets, {"a": 0, "b": 256, "d": 0})
        plan.validate()

    def test_plan_is_deterministic_and_round_trips(self):
        first = plan_memory(planning_graph(), alignment=64)
        second = plan_memory(planning_graph(), alignment=64)
        self.assertEqual(first, second)
        self.assertEqual(MemoryPlan.from_dict(first.to_dict()), first)

    def test_aliases_are_forbidden_or_conservatively_validated(self):
        graph = planning_graph()
        with self.assertRaisesRegex(ValidationError, "aliases are forbidden"):
            plan_memory(graph, aliases={"d": "a"})
        plan = plan_memory(graph, aliases={"d": "a"}, alias_policy="validated")
        self.assertEqual(plan.allocation("d").alias_of, "a")
        self.assertEqual(plan.allocation("d").offset, plan.allocation("a").offset)
        with self.assertRaisesRegex(ValidationError, "overlaps the lifetime"):
            plan_memory(graph, aliases={"b": "a"}, allow_aliases=True)
        with self.assertRaisesRegex(ValidationError, "must be an internal arena"):
            plan_memory(graph, aliases={"y": "d"}, allow_aliases=True)

    def test_rejects_dynamic_or_missing_specs_and_bad_alignment(self):
        dynamic = Graph(
            "dynamic",
            {"x": TensorSpec((None, 4), "float32")},
            ("x",),
            [],
        )
        with self.assertRaisesRegex(ValidationError, "static shape"):
            plan_memory(dynamic)
        graph = Graph(
            "missing",
            {"x": TensorSpec((4,), "float32")},
            ("y",),
            [Node("id", "Identity", ("x",), ("y",))],
        )
        with self.assertRaisesRegex(ValidationError, "no TensorSpec"):
            plan_memory(graph)
        with self.assertRaisesRegex(ValidationError, "power of two"):
            plan_memory(planning_graph(), alignment=96)

    def test_graph_input_can_also_be_an_output(self):
        graph = Graph("passthrough", {"x": TensorSpec((2,), "float32")}, ("x",), [])
        plan = plan_memory(graph)
        value = plan.allocation("x")
        self.assertEqual(value.storage, "external-input")
        self.assertTrue(value.is_input)
        self.assertTrue(value.is_output)
        self.assertEqual(value.last_use, 0)

    def test_internal_reshape_is_a_zero_allocation_view(self):
        graph = reshape_view_graph()
        plan = plan_memory(graph)
        source = plan.allocation("materialized")
        view = plan.allocation("view")

        self.assertEqual(view.view_of, "materialized")
        self.assertIsNone(view.alias_of)
        self.assertEqual(view.offset, source.offset)
        self.assertEqual(source.last_use, 2)
        self.assertEqual(plan.arena_size, source.size_bytes)
        self.assertEqual(plan.peak_live_bytes, source.size_bytes)
        self.assertEqual(MemoryPlan.from_dict(plan.to_dict()), plan)
        plan.validate()

        program = compile_cuda(graph)
        self.assertEqual([launch.node_id for launch in program.launches], ["materialize", "consume"])
        self.assertEqual(
            program.describe()["zero_copy_views"],
            [{"value": "view", "source": "materialized", "byte_offset": 0}],
        )

    def test_reshape_view_never_aliases_public_abi_storage(self):
        source = TensorSpec((2, 3), "float32")
        reshaped = TensorSpec((3, 2), "float32")
        graph = Graph(
            "reshape-input",
            {"x": source},
            ("y",),
            [
                Node("reshape", "Reshape", ("x",), ("reshaped",), {"shape": [3, 2]}),
                Node("consume", "Identity", ("reshaped",), ("y",)),
            ],
            value_specs={"reshaped": reshaped, "y": reshaped},
        )
        plan = plan_memory(graph)
        self.assertIsNone(plan.allocation("reshaped").view_of)
        self.assertEqual(len(compile_cuda(graph).launches), 2)

    def test_contiguous_slice_is_an_offset_view_but_strided_rows_materialize(self):
        source = TensorSpec((2, 3, 4), "float32")
        contiguous = Graph(
            "slice-view",
            {"x": source},
            ("y",),
            [
                Node("materialize", "Identity", ("x",), ("materialized",)),
                Node(
                    "slice",
                    "Slice",
                    ("materialized",),
                    ("view",),
                    {"starts": [1], "ends": [2], "axes": [0], "steps": [1]},
                ),
                Node("consume", "Identity", ("view",), ("y",)),
            ],
        )
        program = compile_cuda(contiguous)
        plan = plan_memory(program.graph)
        source_allocation = plan.allocation("materialized")
        view = plan.allocation("view")
        self.assertEqual([launch.node_id for launch in program.launches], ["materialize", "consume"])
        self.assertEqual(view.view_of, "materialized")
        self.assertEqual(view.offset, source_allocation.offset + 3 * 4 * 4)
        self.assertEqual(plan.arena_size, source_allocation.size_bytes)
        self.assertEqual(
            program.describe()["zero_copy_views"],
            [{"value": "view", "source": "materialized", "byte_offset": 48}],
        )

        non_contiguous = Graph(
            "slice-materialized",
            {"x": source},
            ("y",),
            [
                Node("materialize", "Identity", ("x",), ("materialized",)),
                Node(
                    "slice",
                    "Slice",
                    ("materialized",),
                    ("rows",),
                    {"starts": [1], "ends": [3], "axes": [2], "steps": [1]},
                ),
                Node("consume", "Identity", ("rows",), ("y",)),
            ],
        )
        non_contiguous_program = compile_cuda(non_contiguous)
        self.assertEqual(
            [launch.node_id for launch in non_contiguous_program.launches],
            ["materialize", "slice", "consume"],
        )
        self.assertIsNone(plan_memory(non_contiguous_program.graph).allocation("rows").view_of)

    def test_view_owner_stays_live_while_a_consumer_writes(self):
        source = TensorSpec((2, 3), "float32")
        reshaped = TensorSpec((3, 2), "float32")
        graph = Graph(
            "reshape-view-lifetime",
            {"x": source},
            ("y",),
            [
                Node("materialize", "Identity", ("x",), ("materialized",)),
                Node("reshape", "Reshape", ("materialized",), ("view",), {"shape": [3, 2]}),
                Node("consume", "Identity", ("view",), ("consumed",)),
                Node("output", "Identity", ("consumed",), ("y",)),
            ],
            value_specs={
                "materialized": source,
                "view": reshaped,
                "consumed": reshaped,
                "y": reshaped,
            },
        )
        plan = plan_memory(graph)
        owner = plan.allocation("materialized")
        view = plan.allocation("view")
        consumed = plan.allocation("consumed")
        self.assertEqual(owner.last_use, 2)
        self.assertEqual(view.offset, owner.offset)
        self.assertNotEqual(consumed.offset, owner.offset)
        self.assertEqual(plan.peak_live_bytes, owner.size_bytes + consumed.size_bytes)
        plan.validate()


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.graph = planning_graph()

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name: str = "model.hxp", source: str = "// static CUDA source\n") -> Path:
        return write_model_package(
            self.root / name,
            graph=self.graph,
            program_source=source,
            target="cuda-sm86",
            program_metadata={"entry_points": ["kernel_0"]},
            metadata={"purpose": "unit-test"},
        )

    def test_deterministic_package_layout_and_verified_round_trip(self):
        first = self.write("first.hxp")
        second = self.write("second.hxp")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            self.assertEqual(
                archive.namelist(),
                ["manifest.json", "graph.json", "program.cu", "constants/0000.npy"],
            )
            graph_record = json.loads(archive.read("graph.json"))
            self.assertNotIn("data", graph_record["constants"]["weight"])
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(set(manifest["files"]), {"graph.json", "program.cu", "constants/0000.npy"})
            for name in manifest["files"]:
                self.assertEqual(len(manifest["files"][name]["sha256"]), 64)

        loaded = load_package(first)
        self.assertEqual(loaded.target, "cuda-sm86")
        self.assertEqual(loaded.program_source, "// static CUDA source\n")
        self.assertEqual(loaded.graph.to_json(), self.graph.to_json())
        np.testing.assert_array_equal(loaded.graph.constants["weight"], self.graph.constants["weight"])
        self.assertFalse(loaded.graph.constants["weight"].flags.writeable)
        self.assertEqual(loaded.memory_plan, plan_memory(self.graph))

    def test_source_is_loaded_as_inert_text(self):
        marker = self.root / "must-not-exist"
        source = f"__import__('pathlib').Path({str(marker)!r}).write_text('bad')"
        loaded = load_package(self.write(source=source))
        self.assertEqual(loaded.program_source, source)
        self.assertFalse(marker.exists())

    def test_compiled_program_round_trip_requires_regeneration_match(self):
        program = compile_cuda(self.graph)
        path = write_package(self.root / "compiled.hxp", program)
        rebuilt = rebuild_cuda_program(load_package(path))
        self.assertEqual(rebuilt.source, program.source)
        self.assertEqual(rebuilt.describe()["launches"], program.describe()["launches"])

        forged = write_model_package(
            self.root / "forged.hxp",
            graph=program.graph,
            program_source="// valid UTF-8 but not generated from the graph\n",
            target="cuda-sm86",
            program_metadata=program.describe(),
        )
        with self.assertRaisesRegex(ValidationError, "does not match"):
            rebuild_cuda_program(load_package(forged))

    def test_fused_add_relu_package_is_deterministic_and_rebuildable(self):
        graph = Graph(
            "packaged_add_relu",
            {
                "left": TensorSpec((2, 1, 3), "float32"),
                "right": TensorSpec((1, 4, 1), "float32"),
            },
            ("y",),
            [
                Node("add", "Add", ("left", "right"), ("summed",)),
                Node("relu", "ReLU", ("summed",), ("y",)),
            ],
        )
        program = compile_cuda(graph)
        first = write_package(self.root / "add-relu-first.hxp", program)
        second = write_package(self.root / "add-relu-second.hxp", program)

        self.assertEqual(first.read_bytes(), second.read_bytes())
        loaded = load_package(first)
        self.assertEqual([node.op for node in loaded.graph.nodes], ["AddReLU"])
        rebuilt = rebuild_cuda_program(loaded)
        self.assertEqual(rebuilt.source, program.source)
        self.assertEqual(rebuilt.describe()["launches"], program.describe()["launches"])

    def test_rejects_payload_corruption_and_target_mismatch(self):
        good = self.write()
        corrupt = self.root / "corrupt.hxp"
        rewrite_zip(
            good,
            corrupt,
            lambda members: [
                (name, data + b"tamper" if name == "program.cu" else data)
                for name, data in members
            ],
        )
        with self.assertRaisesRegex(ValidationError, "wrong size|SHA256"):
            load_package(corrupt)
        with self.assertRaisesRegex(ValidationError, "does not match expected"):
            load_package(good, expected_target="cuda-sm90")
        self.assertEqual(load_package(good, expected_target=None).target, "cuda-sm86")

    def test_rejects_path_traversal_duplicate_and_extra_members(self):
        good = self.write()
        traversal = self.root / "traversal.hxp"
        rewrite_zip(good, traversal, lambda members: members + [("../outside", b"bad")])
        with self.assertRaisesRegex(ValidationError, "unsafe member path"):
            load_package(traversal)

        duplicate = self.root / "duplicate.hxp"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rewrite_zip(
                good,
                duplicate,
                lambda members: members + [("program.cu", b"duplicate")],
            )
        with self.assertRaisesRegex(ValidationError, "duplicate member"):
            load_package(duplicate)

        extra = self.root / "extra.hxp"
        rewrite_zip(good, extra, lambda members: members + [("extra.bin", b"bad")])
        with self.assertRaisesRegex(ValidationError, "differs from manifest"):
            load_package(extra)

    def test_rejects_manifest_schema_and_memory_plan_tampering(self):
        good = self.write()

        def mutate_manifest(members, mutation):
            result = []
            for name, data in members:
                if name == "manifest.json":
                    manifest = json.loads(data)
                    mutation(manifest)
                    data = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                result.append((name, data))
            return result

        schema = self.root / "schema.hxp"
        rewrite_zip(good, schema, lambda members: mutate_manifest(members, lambda m: m.update(schema_version=99)))
        with self.assertRaisesRegex(ValidationError, "unsupported package schema"):
            load_package(schema)

        memory = self.root / "memory.hxp"

        def break_memory(manifest):
            allocations = manifest["memory_plan"]["allocations"]
            arena = [item for item in allocations if item["storage"] == "arena"]
            arena[1]["offset"] = arena[0]["offset"]

        rewrite_zip(good, memory, lambda members: mutate_manifest(members, break_memory))
        with self.assertRaisesRegex(ValidationError, "arena_size|overlap in bytes and lifetime"):
            load_package(memory)

    def test_rejects_unsafe_constant_types_and_size_limit(self):
        bad = Graph(
            "object",
            {"x": TensorSpec((1,), "float32")},
            ("x",),
            [],
            constants={"bad": np.asarray([object()], dtype=object)},
        )
        with self.assertRaisesRegex(ValidationError, "object-dtype"):
            write_model_package(self.root / "object.hxp", graph=bad, program_source="", target="cuda-sm86")
        good = self.write()
        with self.assertRaisesRegex(ValidationError, "size limit"):
            load_package(good, max_uncompressed_bytes=10)


if __name__ == "__main__":
    unittest.main()
