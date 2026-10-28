"""Read-only comparison of the two controlled TensorRT inspection exports."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "controlled-build-reports"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(side, suffix):
    return json.loads((REPORTS / f"{side}-build{suffix}.json").read_text())


stock, custom = read("stock", ".layers")["Layers"], read("custom", ".layers")["Layers"]
sb, cb = read("stock", ""), read("custom", "")
sm, cm = ({row["Name"]: row for row in rows} for rows in (stock, custom))
assert len(sm) == len(stock) and len(cm) == len(custom), "ambiguous duplicate layer names"
assert sm.keys() == cm.keys(), "layer membership differs"

# Names select candidate pairs only. Verify every inspection field, and verify
# every renamed edge at all producers/consumers rather than assuming semantics.
renames = {}
name_only_io_changes = []
for name, left in sm.items():
    right = cm[name]
    changed = False
    for direction in ("Inputs", "Outputs"):
        assert len(left.get(direction, [])) == len(right.get(direction, []))
        for lport, rport in zip(left.get(direction, []), right.get(direction, [])):
            assert {k: v for k, v in lport.items() if k != "Name"} == {
                k: v for k, v in rport.items() if k != "Name"
            }, (name, direction, "port descriptor differs")
            if lport["Name"] != rport["Name"]:
                prior = renames.setdefault(lport["Name"], rport["Name"])
                assert prior == rport["Name"], "inconsistent edge rename"
                changed = True
    if changed:
        name_only_io_changes.append(name)
assert len(set(renames.values())) == len(renames), "non-bijective edge renames"


def occurrences(rows, tensor):
    return sorted((layer["Name"], direction, index)
                  for layer in rows for direction in ("Inputs", "Outputs")
                  for index, port in enumerate(layer.get(direction, []))
                  if port["Name"] == tensor)


rename_evidence = []
for old, new in renames.items():
    left, right = occurrences(stock, old), occurrences(custom, new)
    assert left == right, "renamed tensor changes producers or consumers"
    rename_evidence.append({"stock": old, "custom": new,
                           "all_producer_consumer_slots_identical": True,
                           "occurrences": left})

normalized = {}
for name, row in sm.items():
    changed = deepcopy(row)
    for direction in ("Inputs", "Outputs"):
        for port in changed.get(direction, []):
            port["Name"] = renames.get(port["Name"], port["Name"])
    normalized[name] = changed

differences = []
for name, left in normalized.items():
    right = cm[name]
    fields = sorted(k for k in left.keys() | right.keys() if left.get(k) != right.get(k))
    if fields:
        differences.append({"name": name, "different_fields": fields,
                            "stock": {k: left.get(k) for k in fields},
                            "custom": {k: right.get(k) for k in fields}})

msda = []
for name, row in sm.items():
    if row.get("PluginType") != "MultiscaleDeformableAttnPlugin_TRT":
        continue
    other = cm[name]
    assert row["Inputs"] == other["Inputs"] and row["Outputs"] == other["Outputs"]
    msda.append({"name": name, "stock_plugin": [row["PluginType"], row["PluginVersion"]],
                 "custom_plugin": [other["PluginType"], other["PluginVersion"]],
                 "input_output_descriptors_identical_including_tensor_names": True,
                 "inputs": row["Inputs"], "outputs": row["Outputs"]})
assert len(msda) == 18
assert len(differences) == 6 and all(d["different_fields"] == ["PluginType", "PluginVersion"]
                                    for d in differences)

projection_rows = [row for row in stock if "attentions.1/deformable_attention/value_proj" in row["Name"]]
assert all(row == cm[row["Name"]] for row in projection_rows)
fused = [row for row in projection_rows if row["LayerType"] in ("CaskConvolution", "CaskGemmConvolution")]
assert len(fused) == 1
projection = fused[0]
assert all(f"/encoder/layers.{i}/attentions.1/deformable_attention/value_proj/MatMul" in projection["Metadata"]
           for i in range(6))

position_stock = {row["Name"]: index for index, row in enumerate(stock)}
position_custom = {row["Name"]: index for index, row in enumerate(custom)}
moved = [name for name in sm if position_stock[name] != position_custom[name]]
tactic_different = [name for name in sm if any(sm[name].get(k) != cm[name].get(k)
                                             for k in ("TacticName", "TacticValue"))]
assert not tactic_different
settings = ("workspace_limit_bytes", "builder_optimization_level", "tactic_dram_limit_bytes", "max_aux_streams")
assert all(sb[k] == cb[k] for k in settings)
assert sb["timing_cache"]["output"] == cb["timing_cache"]["input"]
assert sb["precision_policy"] == cb["precision_policy"]
assert sb["io"] == cb["io"]

result = {
    "schema_version": 1,
    "scope": "Controlled stock/custom TensorRT inspector comparison; no inference or numerical equivalence claim",
    "method": [
        "Unique layer names are candidate pairing keys, not semantic proof.",
        "Compare every paired inspection field, including metadata, all ordered port descriptors, tactic names/values, weights/bias descriptors, formats, and stream IDs.",
        "Six temporary tensor renames must be bijective and preserve every producer/consumer layer and port index globally.",
        "After only these proven tensor renames, retain and report all remaining field differences.",
    ],
    "sources": {side: {
        "build_report": str(REPORTS / f"{side}-build.json"),
        "build_report_sha256": sha(REPORTS / f"{side}-build.json"),
        "layer_report": str(REPORTS / f"{side}-build.layers.json"),
        "layer_report_sha256": sha(REPORTS / f"{side}-build.layers.json"),
        "engine_sha256": report["engine_sha256"],
        "onnx_sha256": report["onnx_file_sha256"],
    } for side, report in (("stock", sb), ("custom", cb))},
    "summary": {
        "stock_layers": len(stock), "custom_layers": len(custom),
        "layer_membership_identical": True,
        "full_record_identical_after_name_pairing": sum(sm[name] == cm[name] for name in sm),
        "tactic_name_or_value_differences": len(tactic_different),
        "io_shape_dtype_format_location_differences": 0,
        "stream_id_differences": sum(sm[name].get("StreamId") != cm[name].get("StreamId") for name in sm),
        "layer_type_differences": sum(sm[name]["LayerType"] != cm[name]["LayerType"] for name in sm),
        "non_msda_records_identical_after_verified_tensor_renames": True,
        "remaining_changed_records": len(differences),
        "msda_descriptors_identical": len(msda),
        "temporary_tensor_renames": len(renames),
        "layers_with_temporary_tensor_name_changes": len(name_only_io_changes),
        "layers_moved_in_inspection_order": len(moved),
        "moved_layer_types": dict(Counter(sm[name]["LayerType"] for name in moved)),
        "reformat_count": {"stock": sum(row["LayerType"] == "Reformat" for row in stock),
                           "custom": sum(row["LayerType"] == "Reformat" for row in custom)},
    },
    "layer_type_counts": {"stock": dict(Counter(row["LayerType"] for row in stock)),
                          "custom": dict(Counter(row["LayerType"] for row in custom))},
    "build_settings_equal": {key: sb[key] for key in settings},
    "precision_policy_and_engine_io_equal": True,
    "timing_cache": {"stock_output_exactly_equals_custom_input": True,
                     "stock": sb["timing_cache"], "custom": cb["timing_cache"]},
    "spatial_projection": {
        "all_related_records_exactly_equal": True,
        "related_record_count": len(projection_rows),
        "six_value_projections_fused_into_one_convolution": True,
        "fused_projection": {key: projection.get(key) for key in
                             ("LayerType", "TacticName", "TacticValue", "Weights", "Bias", "Kernel", "OutMaps")},
        "input_descriptors": [{k: v for k, v in port.items() if k != "Name"} for port in projection["Inputs"]],
        "output_descriptors": [{k: v for k, v in port.items() if k != "Name"} for port in projection["Outputs"]],
    },
    "msda": msda,
    "remaining_differences_after_verified_tensor_renames": differences,
    "tensor_renames": rename_evidence,
    "reordered_layers": [{"name": name, "stock_position": position_stock[name],
                           "custom_position": position_custom[name],
                           "full_record_identical": sm[name] == cm[name]} for name in moved],
    "limitations": [
        "Inspector descriptors do not expose weight payload bytes, compiled kernel binaries, device addresses, or every internal execution decision.",
        "Matching exposed tactics and descriptors is strong structural evidence, not proof of equal runtime tensor values or bitwise engine outputs.",
        "NonZero and DeviceToShapeHost inspection ordering differs; all their full records and connections match after name pairing.",
        "Runtime A/B/B/A validation remains separate and is not included in this audit.",
    ],
}
output = ROOT / "controlled-tactic-audit.json"
output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(result["summary"], indent=2))
print(output)
