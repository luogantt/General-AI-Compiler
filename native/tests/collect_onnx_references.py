#!/usr/bin/env python3
"""Collect independent ORT references for selected nodes; never execute GAC kernels.

This is an optional diagnostic tool. Its outputs are only used for comparison,
never substituted into the generated native/Vortex model execution.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--node-regex", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(args.model))
    selector = re.compile(args.node_regex)
    selected = [node for node in model.graph.node if selector.search(node.name)]
    if not selected:
        raise ValueError("No source ONNX nodes match --node-regex")
    if any(len(node.output) != 1 for node in selected):
        raise ValueError("Reference-by-node requires single-output ONNX nodes")
    original_outputs = [value.name for value in model.graph.output]
    output_names = list(dict.fromkeys(original_outputs + [node.output[0] for node in selected]))
    # Empty output type is deliberately left for ORT's own inference. We do not
    # copy GAC shapes or types into the reference and thereby validate against it.
    del model.graph.output[:]
    model.graph.output.extend(onnx.helper.make_empty_tensor_value_info(name) for name in output_names)
    diagnostic_model = args.output_dir / "reference-model.onnx"
    onnx.save(model, str(diagnostic_model))
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(diagnostic_model), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    loaded = np.load(args.inputs, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        if len(session.get_inputs()) != 1:
            raise ValueError("Use named .npz for a model with multiple inputs")
        inputs = {session.get_inputs()[0].name: loaded}
    else:
        with loaded:
            inputs = {name: loaded[name] for name in loaded.files}
    start = time.perf_counter()
    values = session.run(output_names, inputs)
    elapsed = time.perf_counter() - start
    by_name = dict(zip(output_names, values))
    np.savez(args.output_dir / "model-reference.npz",
             **{name: by_name[name] for name in original_outputs})
    np.savez(args.output_dir / "node-references.npz",
             **{node.name: by_name[node.output[0]] for node in selected})
    report = {"reference_only": True, "generated_gac_model_executed": False,
              "onnxruntime_version": ort.__version__, "onnx_version": onnx.__version__,
              "threads": args.threads, "reference_execution_wall_seconds": elapsed,
              "model_outputs": original_outputs,
              "nodes": [{"source_node": node.name, "onnx_output": node.output[0],
                         "shape": list(by_name[node.output[0]].shape),
                         "dtype": str(by_name[node.output[0]].dtype)} for node in selected]}
    (args.output_dir / "reference-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
