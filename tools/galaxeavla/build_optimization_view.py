#!/usr/bin/env python3
"""Build an offline LIBERO latency comparison from recorded benchmark JSON."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/validation/ascend/galaxeavla-g05-optimization-910b2-2026-11-16"
DEFAULT_OUTPUT = ROOT / "docs/visualizations/g05-ascend-optimization.html"
TASK = "open the middle drawer of the cabinet"
IDENTITY_FIELDS = {"checkpoint_sha256", "sidecars", "input_files_sha256", "noise_tensor_sha256",
                   "input_ids_sha256", "contract", "weights", "autocast"}
EXPERIMENT_FLAGS = ("context_optimization", "time_conditioning", "ae_attention", "static_vision", "ae_norm", "ae_rope", "functional_mrope", "ae_qknorm_rope")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(row, historical=False):
    keys = {"p50": "median_ms", "p95": "p95_ms", "min": "min_ms", "max": "max_ms", "mean": "mean_ms"} if historical else {k: k for k in ("p50", "p95", "min", "max", "mean")}
    result = {key: float(row[source]) for key, source in keys.items()}
    if not all(math.isfinite(value) and value > 0 for value in result.values()):
        raise ValueError("Latency evidence must contain positive finite statistics")
    if not result["min"] <= result["p50"] <= result["p95"] <= result["max"]:
        raise ValueError("Latency quantiles are inconsistent")
    return result


def backend_label(ae, vlm, ae_options, vlm_options):
    names = {"torchair": "TorchAir", "torchair_aclgraph": "TorchAir ACL Graph", "npugraph_ex": "NPUGraph", "eager": "eager"}
    if ae == "eager" and vlm == "eager":
        return "eager"
    if ae == vlm:
        label = names.get(ae, ae) + " AE + VLM"
    else:
        parts = [names.get(ae, ae) + " AE"] if ae != "eager" else []
        if vlm != "eager":
            parts.append(names.get(vlm, vlm) + " VLM")
        label = " + ".join(parts)
    compiled = [(name, options) for name, backend, options in (("AE", ae, ae_options), ("VLM", vlm, vlm_options)) if backend != "eager"]
    frozen = [name for name, options in compiled if options.get("frozen_parameter") is True]
    if frozen:
        label += " · frozen" if len(frozen) == len(compiled) else " · " + "/".join(frozen) + " frozen"
    single_stream = [name for name, options in compiled if options.get("single_stream") is True]
    if single_stream:
        label += " · " + "/".join(single_stream) + " single_stream"
    return label


def validate_workload(workload, identity, source_revision=None):
    """Reject partial identities and arithmetic that cannot be reconciled."""
    if not isinstance(workload, dict) or not isinstance(workload.get("identity"), dict):
        raise ValueError("Workload requires an explicit comparison identity")
    if not IDENTITY_FIELDS <= set(identity) or not IDENTITY_FIELDS <= set(workload["identity"]):
        raise ValueError("Workload identity is incomplete")
    if workload["identity"] != identity:
        raise ValueError("Workload identity mismatch: complete input/checkpoint/contract identity must match")
    if source_revision and workload.get("source_revision") not in (None, source_revision):
        raise ValueError("Workload source revision mismatch")
    exclusions = workload.get("not_counted")
    if workload.get("diagnostic_requests") != 1 or not isinstance(exclusions, list) or not exclusions or not all(isinstance(item, str) for item in exclusions):
        raise ValueError("Workload must identify one diagnostic request and its excluded arithmetic")
    summary = workload["summary"]
    total = summary["counted_flops"]
    stages, operators = summary["stages"], workload["operators"]
    counts = [total, *(row["counted_flops"] for row in stages.values()), *(row["counted_flops"] for row in operators)]
    if any(type(value) is not int or value < 0 for value in counts) or total <= 0:
        raise ValueError("Workload counts must be nonnegative integers with a positive total")
    if total != sum(row["counted_flops"] for row in stages.values()) or total != sum(row["counted_flops"] for row in operators):
        raise ValueError("Workload stage/operator arithmetic does not reconcile with total")
    for name, row in stages.items():
        if row["counted_flops"] != sum(op["counted_flops"] for op in operators if op["stage"] == name):
            raise ValueError(f"Workload operator arithmetic does not reconcile with stage: {name}")
    return {
        "partial": True, "countedFlops": total,
        "stageFlops": {"prefill": stages.get("prefill", {}).get("counted_flops"), "action": stages.get("fm", {}).get("counted_flops")},
        "allStages": {name: row["counted_flops"] for name, row in stages.items()},
        "operatorRecords": len(operators), "parameters": workload.get("parameters", {}),
        "notCounted": workload["not_counted"], "arithmeticConvention": workload.get("arithmetic_convention"),
        "timingScope": workload.get("timing_scope"), "instrumentationSha256": workload.get("instrumentation_sha256"),
    }


def normalize(path: Path, key: str, label: str | None = None):
    report = json.loads(path.read_text(encoding="utf-8"))
    historical = "timing" in report and "variant" in report
    if historical:
        variant, provenance = report["variant"], report["provenance"]
        if (variant["fm_steps"], variant["resident_weights"], provenance["task"], provenance["episode"], provenance["frame"]) != (10, "bf16", TASK, 0, 0):
            raise ValueError("Historical report does not describe the original LIBERO FM10 case")
        if not report["output"]["all_finite"]:
            raise ValueError("Historical output was nonfinite")
        timings = {stage: stats(report["timing"][source], True) for stage, source in (("prefill", "prefill"), ("action", "action"), ("total", "total"))}
        cold = report.get("setup", {}).get("first_warmup_wall_ms")
        result = {"id": key, "label": label or "RTX 3090 · AE 编译", "short": "3090 历史参考", "historical": True,
                  "device": report["environment"]["gpu"], "timing": timings, "count": provenance["repeats"], "warmup": provenance["warmup"],
                  "backend": "TorchInductor · reduce-overhead", "vlmBackend": "eager", "noiseMode": "native", "noiseLabel": "设备原生 randn",
                  "sourceRevision": None, "sourceHashes": {}, "diffSha256": None, "checkpointSha256": None,
                  "coldMs": cold, "coldLabel": "首次预热墙钟，含编译 / 首次运行", "verification": None,
                  "notes": "用户原始 3090 报告，本轮未在 3090 重跑；历史 seed 数值不能直接作为跨设备噪声参考。"}
    else:
        if report.get("status") != "success":
            raise ValueError(f"Cannot headline unsuccessful report: {path}")
        cfg = report["contract"]
        expected = {"model.model_arch.action_dim": 20, "data.action_size": 32, "model.processor.num_output_cameras": 2,
                    "model.model_arch.fm.num_inference_steps": 10, "model.model_arch.predict_cot": False,
                    "model.model_arch.discrete_action": False, "model.model_arch.continuous_action": True}
        if any(cfg.get(name) != wanted for name, wanted in expected.items()) or report["input"]["task"] != TASK:
            raise ValueError("Expected the real two-camera LIBERO FM10 workload")
        if report["comparison_identity"]["weights"] != "bf16":
            raise ValueError("Comparison requires resident BF16 weights")
        timings = {stage: stats(report["summary_ms"][source]) for stage, source in (("prefill", "prefill_ms"), ("action", "action_ms"), ("total", "total_ms"))}
        backend = report["compilation"].get("name", report["compilation"].get("backend", "unknown"))
        vlm = report.get("vlm_compilation", {}).get("name", "eager")
        noise = report.get("noise_timing")
        if isinstance(noise, str) and "clone" in noise:
            mode = "shared"
        elif isinstance(noise, dict):
            mode = noise["mode"]
        else:
            raise ValueError("Unknown timing noise convention")
        if mode not in {"native", "shared"}:
            raise ValueError(f"Unsupported timing noise mode: {mode}")
        replay = report.get("replay_verification", {})
        cases = dict(replay.get("cases", {}))
        context_cases = report.get("context_verification", {}).get("cases", {})
        for name, row in context_cases.items():
            if not all(row.get("encoded_context", {}).get(key) is True for key in ("input_ids", "attention_mask")):
                raise ValueError("Context optimization has mismatched encoded inputs")
            cases[name] = row["action"]
        verification = None
        if cases:
            if any(row.get("valid_action_dimensions") != [10, 11, 12, 13, 14, 15, 19] or row.get("sample_count") != 224 for row in cases.values()):
                raise ValueError("Replay comparison must cover 32 x 7 valid LIBERO action values")
            thresholds = report["numerical_thresholds"]
            verification = {
                "cases": cases, "thresholds": thresholds,
                "contextCases": list(context_cases),
                "complete": all(name in cases for name in ("cold_A", "first_B", "replay_A", "replay_B")),
                "distinguishable": replay.get("reference_inputs_distinguishable", False),
                "maxAbs": max(row["max_abs_error"] for row in cases.values()),
                "maxRelative": max(row["relative_l2"] for row in cases.values()),
                "maxRmse": max(row["rmse"] for row in cases.values()),
                "minSignAgreement": min(row["sign_agreement"] for row in cases.values()) if all("sign_agreement" in row for row in cases.values()) else None,
                "minGripperSignAgreement": min(row["gripper_sign_agreement"] for row in cases.values()) if all("gripper_sign_agreement" in row for row in cases.values()) else None,
                "passed": all(row["all_finite"] and row["max_abs_error"] <= thresholds["max_abs_error"] and row["relative_l2"] <= thresholds["max_relative_l2"] and row.get("padded_dimensions_max_abs_candidate", 0) == 0 for row in cases.values()),
                "scope": replay.get("scope", "") + ("; " + report["context_verification"]["scope"] if context_cases else ""),
            }
        source = report.get("source", {})
        cold = replay.get("timings_ms", {}).get("cold_A", {}).get("total_ms")
        ae_options = report["compilation"].get("options", {})
        vlm_options = report.get("vlm_compilation", {}).get("options", {})
        label_text = backend_label(backend, vlm, ae_options, vlm_options)
        algorithm = report.get("gdn_algorithm", {}).get("name", "original")
        context_optimized = report.get("context_optimization", {}).get("enabled", False)
        static_vision = report.get("static_vision", {}).get("enabled", False)
        ae_norm = report.get("ae_norm", {}).get("enabled", False)
        ae_qknorm_rope = report.get("ae_qknorm_rope", {}).get("enabled", False)
        functional_mrope = report.get("functional_mrope", {}).get("enabled", False)
        ae_rope = report.get("ae_rope", {}).get("mode", "sliced") if report.get("ae_rope", {}).get("enabled", False) else "eager"
        if ae_rope not in {"eager", "sliced", "matrix"}:
            raise ValueError(f"Unsupported AE RoPE mode: {ae_rope}")
        if ae_qknorm_rope and ae_rope != "eager":
            raise ValueError("Ascend C QK fusion and standalone AE RoPE cannot both be enabled")
        if algorithm != "original":
            label_text += " · " + {"gdn-chunk-fp32-triangular-solve": "GDN 三角解",
                "gdn-chunk-fp32-triton-sequential-row-inverse": "GDN Triton"}.get(algorithm, algorithm)
        if any(report.get(key, {}).get("enabled", False) for key in EXPERIMENT_FLAGS):
            if not {"state_C_1", "state_C_2", "task_D_1", "task_D_2"} <= set(context_cases):
                raise ValueError("Context optimization requires state/task replay verification")
        if static_vision:
            if not {"camera_E_1", "camera_E_2", "wrist_F_1", "wrist_F_2"} <= set(context_cases):
                raise ValueError("Static vision requires independent camera/wrist replay verification")
            label_text += " · 固定几何 Vision graph"
        if context_optimized:
            label_text += " · 移除无用编码"
        if report.get("ae_attention", {}).get("enabled", False):
            label_text += " · " + report["ae_attention"]["backend"].upper() + " 融合注意力"
        if report.get("time_conditioning", {}).get("enabled", False):
            label_text += " · 时间条件预计算 (" + report["time_conditioning"].get("lookup_mode", "exact") + ")"
        if ae_norm:
            label_text += " · NPU Adaptive RMSNorm (FP32)" if ae_qknorm_rope else " · NPU RMSNorm (FP32)"
        if ae_rope != "eager":
            label_text += " · NPU RoPE " + {"sliced": "切片", "matrix": "矩阵"}[ae_rope] + " (FP32)"
        if functional_mrope:
            label_text += " · 函数式 MRoPE"
        if ae_qknorm_rope:
            label_text += " · Ascend C QK Norm+RoPE+布局融合 (UB)"
        if mode == "native":
            label_text += " · native RNG"
        if path.parent.name == "libero-bf16-native-matrix-control-20261117":
            if ae_rope != "matrix" or ae_qknorm_rope:
                raise ValueError("The named same-round matrix control must use standalone matrix RoPE")
            label_text += " · 同轮 matrix 对照 (2026-11-17)"
        result = {"id": key, "label": label or "910B2 · " + label_text, "short": label_text, "historical": False,
                  "device": report["environment"].get("device_name", "Ascend 910B2"), "timing": timings, "count": len(report["samples"]), "warmup": len(report["warmup"]),
                  "backend": backend, "vlmBackend": vlm, "aeOptions": ae_options, "vlmOptions": vlm_options,
                  "contextOptimized": context_optimized,
                  "staticVision": static_vision,
                  "aeNorm": ae_norm, "aeNormOperator": report.get("ae_norm", {}).get("operator"), "aeRope": ae_rope,
                  "aeQkNormRope": ae_qknorm_rope,
                  "aeQkNormRopeOperator": report.get("ae_qknorm_rope", {}).get("operator"),
                  "aeQkNormRopePrecision": report.get("ae_qknorm_rope", {}).get("precision"),
                  "aeQkNormRopeScope": report.get("ae_qknorm_rope", {}).get("scope"),
                  "aeQkNormRopeLibrarySha256": report.get("ae_qknorm_rope", {}).get("library_sha256"),
                  "functionalMrope": functional_mrope, "functionalMropeScope": report.get("functional_mrope", {}).get("scope"),
                  "singleStreamScopes": [name for name, options in (("AE", ae_options), ("VLM", vlm_options)) if options.get("single_stream") is True],
                  "noiseMode": mode, "noiseLabel": "固定共享噪声 clone" if mode == "shared" else "设备原生 randn",
                  "sourceRevision": report["source_revision"], "sourceHashes": source.get("key_files_sha256", {}),
                  "diffSha256": source.get("tree_state", {}).get("tracked_diff_sha256"),
                  "checkpointSha256": report["comparison_identity"]["checkpoint_sha256"], "coldMs": cold,
                  "coldLabel": "首次 selected 请求，含编译 / 首次运行" if cold is not None else "未单独记录编译冷请求" if backend != "eager" else "eager，无编译冷启动",
                  "verification": verification,
                  "notes": "同设备 eager 数值回归；未进行闭环机器人任务成功率评估。跨 3090 共享噪声对照尚未提供。"}
    result["reportPath"] = path.resolve().relative_to(ROOT).as_posix() if path.resolve().is_relative_to(ROOT) else path.name
    result["reportSha256"] = digest(path)
    return result, report


def make_data(paths, workload_paths=()):
    configs, raw = [], {}
    for index, path in enumerate(paths):
        key = ("reference", "eager", "torchair")[index] if index < 3 else f"extra-{index - 2}"
        config, report = normalize(path, key)
        configs.append(config)
        raw[key] = {"path": config["reportPath"], "sha256": config["reportSha256"], "report": report}
    identity = raw["eager"]["report"]["comparison_identity"]
    for key, evidence in raw.items():
        if key == "reference":
            continue
        candidate = evidence["report"]["comparison_identity"]
        for field in ("checkpoint_sha256", "input_files_sha256", "contract", "weights"):
            if candidate[field] != identity[field]:
                raise ValueError(f"Comparison evidence differs in {field}: {key}")
    candidates, external_workloads = [], []
    for key, evidence in raw.items():
        report = evidence["report"]
        if report.get("workload") is not None:
            workload = report["workload"]
            validate_workload(workload, report["comparison_identity"], report.get("source_revision"))
            candidates.append((key, evidence["path"] + "#workload", workload))
    for path in workload_paths:
        workload = json.loads(path.read_text(encoding="utf-8"))
        identity = workload.get("identity", {})
        validate_workload(workload, identity)
        matches = [row for row in raw.values() if row["report"].get("comparison_identity") == identity]
        if not matches:
            raise ValueError(f"Explicit workload matches no report identity: {path}")
        label = path.resolve().relative_to(ROOT).as_posix() if path.resolve().is_relative_to(ROOT) else path.name
        candidates.append((None, label, workload))
        external_workloads.append({"path": label, "sha256": digest(path), "workload": workload})
    baselines = [row for row in configs if not row["historical"] and row["backend"] == "eager"
                 and row["vlmBackend"] == "eager"
                 and raw[row["id"]]["report"].get("gdn_algorithm", {}).get("name", "original") == "original"
                 and not any(raw[row["id"]]["report"].get(key, {}).get("enabled", False)
                             for key in EXPERIMENT_FLAGS)]
    if not baselines:
        raise ValueError("Comparison requires an unmodified eager baseline")
    for config in configs:
        median = config["timing"]["total"]["p50"]
        matching_baselines = [row for row in baselines if row["noiseMode"] == config["noiseMode"]]
        baseline = matching_baselines[0] if matching_baselines else baselines[0]
        eager = baseline["timing"]["total"]["p50"]
        config.update(speedup=eager / median, gapMs=median - 120., targetRatio=median / 120.,
                      speedupBaselineMs=eager, speedupBaselineId=baseline["id"],
                      speedupBaselineNoise=baseline["noiseMode"], speedupSameNoise=bool(matching_baselines))
        config["workload"] = None
        report = raw[config["id"]]["report"]
        matching = [(key, label, workload) for key, label, workload in candidates
                    if workload["identity"] == report.get("comparison_identity")
                    and workload.get("execution", {}).get("context_optimization", False)
                        == report.get("context_optimization", {}).get("enabled", False)
                    and workload.get("execution", {}).get("time_conditioning", False)
                        == report.get("time_conditioning", {}).get("enabled", False)
                    and workload.get("execution", {}).get("static_vision", False)
                        == report.get("static_vision", {}).get("enabled", False)
                    and all(workload.get("execution", {}).get(key, False)
                            == report.get(key, {}).get("enabled", False) for key in ("ae_norm", "ae_rope", "functional_mrope", "ae_qknorm_rope"))]
        matching.sort(key=lambda entry: entry[0] != config["id"])
        if matching:
            first_summary = matching[0][2]["summary"]
            if any(entry[2]["summary"] != first_summary for entry in matching[1:]):
                raise ValueError(f"Conflicting workload counts for identical input: {config['id']}")
            owner, label, workload = matching[0]
            config["workload"] = {**validate_workload(workload, report["comparison_identity"], report.get("source_revision")),
                                  "sourceLabel": label, "sameReport": owner == config["id"], "identityMatched": True}
    best = min((row for row in configs if not row["historical"]), key=lambda row: row["timing"]["total"]["p50"])
    return {"schema_version": 1, "targetMs": 120., "defaultId": best["id"], "configs": configs,
            "workload": {"task": TASK, "cameras": 2, "imageSize": [256, 256], "prefixTokens": 159, "imageTokens": 128,
                         "actionShape": [1, 32, 20], "validDimensions": [10, 11, 12, 13, 14, 15, 19], "fmSteps": 10, "flops": None},
            "evidence": {"scope": "Original benchmark reports; historical 3090 report was not rerun", "reports": raw,
                         "external_workloads": external_workloads}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=EVIDENCE / "reference-3090/bf16resident_steps10_compileae.json")
    parser.add_argument("--eager", type=Path, default=EVIDENCE / "libero-bf16-eager/report.json")
    parser.add_argument("--compiled", type=Path, default=EVIDENCE / "libero-bf16-torchair-ae/report.json")
    parser.add_argument("--best-report", type=Path, action="append", default=[], help="Append a successful, matching LIBERO report; repeatable")
    parser.add_argument("--workload", type=Path, action="append", default=[], help="Optional partial-workload JSON, shown only for a fully matching report identity")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    data = make_data([args.reference, args.eager, args.compiled, *args.best_report], args.workload)
    template = Path(__file__).with_name("optimization_view.template.html").read_text(encoding="utf-8")
    if template.count("__OPTIMIZATION_DATA__") != 1:
        raise ValueError("Expected one embedded data placeholder")
    html = template.replace("__OPTIMIZATION_DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8", newline="\n")
    evidence_path = args.output.with_suffix(".evidence.json")
    evidence_path.write_text(json.dumps(data["evidence"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "evidence": str(evidence_path), "configs": len(data["configs"]), "selected": data["defaultId"]}))


if __name__ == "__main__":
    main()
