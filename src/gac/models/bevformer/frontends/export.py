"""Offline BEVFormer-tiny export contract and safe config inspection.

This module intentionally has no dependency on PyTorch, MMCV, MMDetection, or
ONNX.  It freezes the tensor-only ABI that a future framework exporter must
produce and extracts the relevant static profile from ``bevformer_tiny.py`` by
parsing a deliberately small, non-executable subset of Python.

Nothing in this module exports a PyTorch module or creates an ONNX model.  The
manifest records that limitation explicitly so an offline contract cannot be
mistaken for a completed framework export.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


MANIFEST_FORMAT = "gac.bevformer.step-manifest"
MANIFEST_SCHEMA_VERSION = 1
PROFILE_NAME = "bevformer-tiny-nuscenes-b1-cam6-480x800-bev50-fp16-sm86-v1"
ENTRYPOINT = "bevformer_tiny_step"

# This is the insertion order used by the BEVFormer snapshot's nuScenes data
# converter.  The order is part of the ABI because lidar2img and images must
# refer to the same camera at every index.
NUSCENES_CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

_DTYPES = {"float16", "float32", "int32"}
_MAX_AST_NODES = 100_000
_MAX_CONTAINER_ITEMS = 100_000
_MAX_STRING_CHARS = 1_000_000


class BevFormerExportError(ValueError):
    """A config, manifest, or tensor value violates the offline contract."""

    def __init__(
        self, message: str, *, audit_report: PythonAuditReport | None = None
    ) -> None:
        super().__init__(message)
        self.audit_report = audit_report


def _strict_keys(
    value: Mapping[str, Any], expected: set[str], *, where: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise BevFormerExportError(f"{where} has " + "; ".join(details))


def _string(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise BevFormerExportError(f"{where} must be a non-empty string")
    return value


def _integer(value: Any, *, where: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BevFormerExportError(f"{where} must be an integer >= {minimum}")
    return value


def _number(value: Any, *, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BevFormerExportError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise BevFormerExportError(f"{where} must be a finite number")
    return result


def _mapping(value: Any, *, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise BevFormerExportError(f"{where} must be a string-keyed object")
    return value


def _array(value: Any, *, where: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise BevFormerExportError(f"{where} must be an array")
    return value


@dataclass(frozen=True, slots=True)
class AuditFinding:
    """One construct that the static config reader will not execute."""

    code: str
    message: str
    line: int
    column: int
    node_type: str

    def __post_init__(self) -> None:
        _string(self.code, where="audit finding code")
        _string(self.message, where="audit finding message")
        _integer(self.line, where="audit finding line")
        _integer(self.column, where="audit finding column", minimum=0)
        _string(self.node_type, where="audit finding node_type")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "node_type": self.node_type,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> AuditFinding:
        raw = _mapping(raw, where="audit finding")
        _strict_keys(
            raw,
            {"code", "message", "line", "column", "node_type"},
            where="audit finding",
        )
        return cls(
            code=_string(raw["code"], where="audit finding code"),
            message=_string(raw["message"], where="audit finding message"),
            line=_integer(raw["line"], where="audit finding line"),
            column=_integer(
                raw["column"], where="audit finding column", minimum=0
            ),
            node_type=_string(raw["node_type"], where="audit finding node_type"),
        )


@dataclass(frozen=True, slots=True)
class PythonAuditReport:
    """Result of checking a Python file against the static config subset."""

    source: str
    parser: str = "python-ast-static-v1"
    findings: tuple[AuditFinding, ...] = ()

    def __post_init__(self) -> None:
        _string(self.source, where="audit source")
        if self.parser != "python-ast-static-v1":
            raise BevFormerExportError("unsupported audit parser")
        if not isinstance(self.findings, tuple) or not all(
            isinstance(item, AuditFinding) for item in self.findings
        ):
            raise BevFormerExportError("audit findings must be AuditFinding values")

    @property
    def safe_for_static_config(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "parser": self.parser,
            "safe_for_static_config": self.safe_for_static_config,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PythonAuditReport:
        raw = _mapping(raw, where="audit report")
        _strict_keys(
            raw,
            {"source", "parser", "safe_for_static_config", "findings"},
            where="audit report",
        )
        safe = raw["safe_for_static_config"]
        if not isinstance(safe, bool):
            raise BevFormerExportError(
                "audit safe_for_static_config must be a boolean"
            )
        findings_raw = _array(raw["findings"], where="audit findings")
        result = cls(
            source=_string(raw["source"], where="audit source"),
            parser=_string(raw["parser"], where="audit parser"),
            findings=tuple(
                AuditFinding.from_dict(_mapping(item, where="audit finding"))
                for item in findings_raw
            ),
        )
        if safe != result.safe_for_static_config:
            raise BevFormerExportError(
                "audit safe_for_static_config disagrees with findings"
            )
        return result


def _finding(code: str, message: str, node: ast.AST) -> AuditFinding:
    return AuditFinding(
        code=code,
        message=message,
        line=max(1, int(getattr(node, "lineno", 1))),
        column=max(0, int(getattr(node, "col_offset", 0))),
        node_type=type(node).__name__,
    )


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _audit_expression(node: ast.AST, findings: list[AuditFinding]) -> None:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (str, int, float, bool, type(None))):
            findings.append(
                _finding("UNSUPPORTED_LITERAL", "literal type is not supported", node)
            )
        return
    if isinstance(node, ast.Name):
        return
    if isinstance(node, (ast.List, ast.Tuple)):
        for item in node.elts:
            _audit_expression(item, findings)
        return
    if isinstance(node, ast.Dict):
        for key, value in zip(node.keys, node.values, strict=True):
            if key is not None:
                _audit_expression(key, findings)
            _audit_expression(value, findings)
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            findings.append(
                _finding(
                    "UNSUPPORTED_OPERATOR",
                    f"operator {type(node.op).__name__} is not in the static subset",
                    node,
                )
            )
            return
        _audit_expression(node.left, findings)
        _audit_expression(node.right, findings)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            findings.append(
                _finding(
                    "UNSUPPORTED_OPERATOR",
                    f"operator {type(node.op).__name__} is not in the static subset",
                    node,
                )
            )
            return
        _audit_expression(node.operand, findings)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id != "dict":
            findings.append(
                _finding(
                    "UNSAFE_CALL",
                    "only the literal-building dict(...) call is supported",
                    node,
                )
            )
            return
        if len(node.args) > 1:
            findings.append(
                _finding(
                    "UNSUPPORTED_DICT_CALL",
                    "dict(...) accepts at most one mapping in the static subset",
                    node,
                )
            )
        for arg in node.args:
            _audit_expression(arg, findings)
        for keyword in node.keywords:
            _audit_expression(keyword.value, findings)
        return
    findings.append(
        _finding(
            "DYNAMIC_EXPRESSION",
            f"{type(node).__name__} is not in the static config subset",
            node,
        )
    )


def _parse_and_audit(
    source: str, *, source_name: str
) -> tuple[ast.Module | None, PythonAuditReport]:
    if not isinstance(source, str):
        raise BevFormerExportError("Python source must be text")
    try:
        tree = ast.parse(source, filename=source_name, mode="exec")
    except SyntaxError as exc:
        finding = AuditFinding(
            code="SYNTAX_ERROR",
            message=exc.msg,
            line=max(1, int(exc.lineno or 1)),
            column=max(0, int((exc.offset or 1) - 1)),
            node_type="SyntaxError",
        )
        return None, PythonAuditReport(source_name, findings=(finding,))

    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > _MAX_AST_NODES:
        finding = AuditFinding(
            code="SOURCE_TOO_COMPLEX",
            message=f"AST contains {node_count} nodes; limit is {_MAX_AST_NODES}",
            line=1,
            column=0,
            node_type="Module",
        )
        return tree, PythonAuditReport(source_name, findings=(finding,))

    findings: list[AuditFinding] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(
                statement.targets[0], ast.Name
            ):
                findings.append(
                    _finding(
                        "DYNAMIC_ASSIGNMENT",
                        "only assignment to one plain name is supported",
                        statement,
                    )
                )
            _audit_expression(statement.value, findings)
        elif isinstance(statement, ast.AnnAssign):
            if not isinstance(statement.target, ast.Name) or statement.value is None:
                findings.append(
                    _finding(
                        "DYNAMIC_ASSIGNMENT",
                        "only initialized assignment to a plain name is supported",
                        statement,
                    )
                )
            else:
                _audit_expression(statement.value, findings)
        elif isinstance(statement, ast.Expr) and isinstance(
            statement.value, ast.Constant
        ) and isinstance(statement.value.value, str):
            # Module docstrings are inert.
            continue
        else:
            category = (
                "UNSAFE_IMPORT"
                if isinstance(statement, (ast.Import, ast.ImportFrom))
                else "DYNAMIC_STATEMENT"
            )
            findings.append(
                _finding(
                    category,
                    f"top-level {type(statement).__name__} is not supported",
                    statement,
                )
            )
    return tree, PythonAuditReport(source_name, findings=tuple(findings))


def audit_python_source(
    source: str, *, source_name: str = "<string>"
) -> PythonAuditReport:
    """Audit *source* without executing it or importing any of its dependencies."""

    tree, report = _parse_and_audit(source, source_name=source_name)
    if tree is None or report.findings:
        return report
    _, evaluation_findings = _evaluate_assignments(tree)
    if not evaluation_findings:
        return report
    return replace(report, findings=evaluation_findings)


def audit_python_file(path: str | Path) -> PythonAuditReport:
    """Read and audit a UTF-8 Python config without importing the file."""

    config_path = Path(path)
    try:
        source = config_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise BevFormerExportError(f"cannot read {config_path}: {exc}") from exc
    return audit_python_source(source, source_name=config_path.as_posix())


class _StaticEvalFailure(Exception):
    def __init__(self, code: str, message: str, node: ast.AST) -> None:
        super().__init__(message)
        self.code = code
        self.node = node


def _bounded(value: Any, node: ast.AST) -> Any:
    if isinstance(value, str) and len(value) > _MAX_STRING_CHARS:
        raise _StaticEvalFailure(
            "VALUE_TOO_LARGE", "static string exceeds size limit", node
        )
    if isinstance(value, (list, tuple, dict)) and len(value) > _MAX_CONTAINER_ITEMS:
        raise _StaticEvalFailure(
            "VALUE_TOO_LARGE", "static container exceeds size limit", node
        )
    return value


def _eval_expression(node: ast.AST, symbols: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return _bounded(node.value, node)
    if isinstance(node, ast.Name):
        if node.id not in symbols:
            raise _StaticEvalFailure(
                "UNRESOLVED_NAME", f"name {node.id!r} is not defined earlier", node
            )
        return symbols[node.id]
    if isinstance(node, ast.List):
        return _bounded([_eval_expression(v, symbols) for v in node.elts], node)
    if isinstance(node, ast.Tuple):
        return _bounded(tuple(_eval_expression(v, symbols) for v in node.elts), node)
    if isinstance(node, ast.Dict):
        result: dict[Any, Any] = {}
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            value = _eval_expression(value_node, symbols)
            if key_node is None:
                expanded = _mapping(value, where="static ** mapping")
                result.update(expanded)
                continue
            key = _eval_expression(key_node, symbols)
            try:
                hash(key)
            except TypeError as exc:
                raise _StaticEvalFailure(
                    "INVALID_DICT_KEY", "static dictionary key is not hashable", key_node
                ) from exc
            result[key] = value
        return _bounded(result, node)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_expression(node.operand, symbols)
        if isinstance(operand, bool) or not isinstance(operand, (int, float)):
            raise _StaticEvalFailure(
                "INVALID_STATIC_OPERATOR", "unary operator requires a number", node
            )
        return +operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        left = _eval_expression(node.left, symbols)
        right = _eval_expression(node.right, symbols)
        try:
            if isinstance(node.op, ast.Add):
                value = left + right
            elif isinstance(node.op, ast.Sub):
                value = left - right
            elif isinstance(node.op, ast.Mult):
                value = left * right
            elif isinstance(node.op, ast.Div):
                value = left / right
            elif isinstance(node.op, ast.FloorDiv):
                value = left // right
            elif isinstance(node.op, ast.Mod):
                value = left % right
            else:  # guarded by the audit, retained for direct robustness
                raise TypeError(type(node.op).__name__)
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise _StaticEvalFailure(
                "INVALID_STATIC_OPERATOR", f"static operator failed: {exc}", node
            ) from exc
        return _bounded(value, node)
    if isinstance(node, ast.Call):
        result: dict[str, Any] = {}
        if node.args:
            result.update(
                _mapping(
                    _eval_expression(node.args[0], symbols),
                    where="dict positional argument",
                )
            )
        for keyword in node.keywords:
            value = _eval_expression(keyword.value, symbols)
            if keyword.arg is None:
                result.update(_mapping(value, where="dict ** argument"))
            else:
                result[keyword.arg] = value
        return _bounded(result, node)
    raise _StaticEvalFailure(
        "DYNAMIC_EXPRESSION",
        f"{type(node).__name__} is not in the static config subset",
        node,
    )


def _evaluate_assignments(
    tree: ast.Module,
) -> tuple[dict[str, Any], tuple[AuditFinding, ...]]:
    symbols: dict[str, Any] = {}
    findings: list[AuditFinding] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(
            statement.targets[0], ast.Name
        ):
            target = statement.targets[0]
            value_node = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ) and statement.value is not None:
            target = statement.target
            value_node = statement.value
        else:
            continue
        try:
            symbols[target.id] = _eval_expression(value_node, symbols)
        except (BevFormerExportError, _StaticEvalFailure) as exc:
            if isinstance(exc, _StaticEvalFailure):
                findings.append(_finding(exc.code, str(exc), exc.node))
            else:
                findings.append(
                    _finding("INVALID_STATIC_VALUE", str(exc), value_node)
                )
    return symbols, tuple(findings)


def _at(value: Any, *path: str | int) -> Any:
    current = value
    rendered: list[str] = []
    for part in path:
        rendered.append(str(part))
        if isinstance(part, str):
            if not isinstance(current, Mapping) or part not in current:
                raise BevFormerExportError(
                    "missing static config value " + ".".join(rendered)
                )
            current = current[part]
        else:
            if (
                isinstance(current, (str, bytes, bytearray))
                or not isinstance(current, Sequence)
                or part < 0
                or part >= len(current)
            ):
                raise BevFormerExportError(
                    "missing static config value " + ".".join(rendered)
                )
            current = current[part]
    return current


def _find_typed(value: Any, type_name: str) -> tuple[Mapping[str, Any], ...]:
    found: list[Mapping[str, Any]] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            if item.get("type") == type_name:
                found.append(item)
            for child in item.values():
                visit(child)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for child in item:
                visit(child)

    visit(value)
    return tuple(found)


def _one_typed(value: Any, type_name: str) -> Mapping[str, Any]:
    found = _find_typed(value, type_name)
    if len(found) != 1:
        raise BevFormerExportError(
            f"expected one {type_name!r} config block, found {len(found)}"
        )
    return found[0]


@dataclass(frozen=True, slots=True)
class BevFormerTinyStaticConfig:
    """The frozen BEVFormer-tiny values used to derive the step ABI."""

    batch: int
    cameras: int
    image_channels: int
    image_content_hw: tuple[int, int]
    image_input_hw: tuple[int, int]
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]
    image_to_rgb: bool
    backbone_depth: int
    feature_shapes: tuple[tuple[int, int], ...]
    embed_dim: int
    bev_hw: tuple[int, int]
    encoder_layers: int
    decoder_layers: int
    object_queries: int
    class_names: tuple[str, ...]
    attention_heads: int
    sampling_points: int
    training_queue_length: int
    can_bus_dim: int
    bbox_code_size: int

    def __post_init__(self) -> None:
        integer_fields = (
            "batch",
            "cameras",
            "image_channels",
            "backbone_depth",
            "embed_dim",
            "encoder_layers",
            "decoder_layers",
            "object_queries",
            "attention_heads",
            "sampling_points",
            "training_queue_length",
            "can_bus_dim",
            "bbox_code_size",
        )
        for name in integer_fields:
            _integer(getattr(self, name), where=f"configuration {name}")
        for name in ("image_content_hw", "image_input_hw", "bev_hw"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or len(value) != 2:
                raise BevFormerExportError(f"configuration {name} must be [H, W]")
            for dim in value:
                _integer(dim, where=f"configuration {name} dimension")
        if not isinstance(self.feature_shapes, tuple) or not self.feature_shapes:
            raise BevFormerExportError("configuration feature_shapes cannot be empty")
        for shape in self.feature_shapes:
            if not isinstance(shape, tuple) or len(shape) != 2:
                raise BevFormerExportError("each feature shape must be [H, W]")
            for dim in shape:
                _integer(dim, where="feature shape dimension")
        for name in ("image_mean", "image_std"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or len(values) != 3:
                raise BevFormerExportError(f"configuration {name} must have 3 values")
            for value in values:
                _number(value, where=f"configuration {name}")
        if any(value <= 0 for value in self.image_std):
            raise BevFormerExportError("configuration image_std values must be positive")
        if not isinstance(self.image_to_rgb, bool):
            raise BevFormerExportError("configuration image_to_rgb must be boolean")
        if not isinstance(self.class_names, tuple) or not self.class_names:
            raise BevFormerExportError("configuration class_names cannot be empty")
        if len(set(self.class_names)) != len(self.class_names) or not all(
            isinstance(name, str) and name for name in self.class_names
        ):
            raise BevFormerExportError(
                "configuration class_names must be unique non-empty strings"
            )
        if self.embed_dim % self.attention_heads:
            raise BevFormerExportError(
                "configuration embed_dim must be divisible by attention_heads"
            )

    @property
    def bev_queries(self) -> int:
        return self.bev_hw[0] * self.bev_hw[1]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def validate_tiny_profile(self) -> None:
        """Reject a config that differs from the first frozen compiler profile."""

        expected: dict[str, Any] = {
            "batch": 1,
            "cameras": 6,
            "image_channels": 3,
            "image_content_hw": (450, 800),
            "image_input_hw": (480, 800),
            "image_mean": (123.675, 116.28, 103.53),
            "image_std": (58.395, 57.12, 57.375),
            "image_to_rgb": True,
            "backbone_depth": 50,
            "feature_shapes": ((15, 25),),
            "embed_dim": 256,
            "bev_hw": (50, 50),
            "encoder_layers": 3,
            "decoder_layers": 6,
            "object_queries": 900,
            "attention_heads": 8,
            "sampling_points": 8,
            "training_queue_length": 3,
            "can_bus_dim": 18,
            "bbox_code_size": 10,
            "num_classes": 10,
        }
        mismatches = [
            f"{name}={getattr(self, name)!r} (expected {wanted!r})"
            for name, wanted in expected.items()
            if getattr(self, name) != wanted
        ]
        if mismatches:
            raise BevFormerExportError(
                "config does not match the frozen BEVFormer-tiny profile: "
                + ", ".join(mismatches)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch": self.batch,
            "cameras": self.cameras,
            "image_channels": self.image_channels,
            "image_content_hw": list(self.image_content_hw),
            "image_input_hw": list(self.image_input_hw),
            "image_mean": list(self.image_mean),
            "image_std": list(self.image_std),
            "image_to_rgb": self.image_to_rgb,
            "backbone_depth": self.backbone_depth,
            "feature_shapes": [list(shape) for shape in self.feature_shapes],
            "embed_dim": self.embed_dim,
            "bev_hw": list(self.bev_hw),
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "object_queries": self.object_queries,
            "class_names": list(self.class_names),
            "attention_heads": self.attention_heads,
            "sampling_points": self.sampling_points,
            "training_queue_length": self.training_queue_length,
            "can_bus_dim": self.can_bus_dim,
            "bbox_code_size": self.bbox_code_size,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> BevFormerTinyStaticConfig:
        raw = _mapping(raw, where="configuration")
        keys = {
            "batch",
            "cameras",
            "image_channels",
            "image_content_hw",
            "image_input_hw",
            "image_mean",
            "image_std",
            "image_to_rgb",
            "backbone_depth",
            "feature_shapes",
            "embed_dim",
            "bev_hw",
            "encoder_layers",
            "decoder_layers",
            "object_queries",
            "class_names",
            "attention_heads",
            "sampling_points",
            "training_queue_length",
            "can_bus_dim",
            "bbox_code_size",
        }
        _strict_keys(raw, keys, where="configuration")

        def pair(name: str) -> tuple[int, int]:
            values = _array(raw[name], where=f"configuration {name}")
            if len(values) != 2:
                raise BevFormerExportError(f"configuration {name} must be [H, W]")
            return (
                _integer(values[0], where=f"configuration {name}[0]"),
                _integer(values[1], where=f"configuration {name}[1]"),
            )

        def triple(name: str) -> tuple[float, float, float]:
            values = _array(raw[name], where=f"configuration {name}")
            if len(values) != 3:
                raise BevFormerExportError(f"configuration {name} must have 3 values")
            return tuple(
                _number(value, where=f"configuration {name}") for value in values
            )  # type: ignore[return-value]

        feature_shapes_raw = _array(
            raw["feature_shapes"], where="configuration feature_shapes"
        )
        class_names_raw = _array(
            raw["class_names"], where="configuration class_names"
        )
        image_to_rgb = raw["image_to_rgb"]
        if not isinstance(image_to_rgb, bool):
            raise BevFormerExportError("configuration image_to_rgb must be boolean")
        result = cls(
            batch=_integer(raw["batch"], where="configuration batch"),
            cameras=_integer(raw["cameras"], where="configuration cameras"),
            image_channels=_integer(
                raw["image_channels"], where="configuration image_channels"
            ),
            image_content_hw=pair("image_content_hw"),
            image_input_hw=pair("image_input_hw"),
            image_mean=triple("image_mean"),
            image_std=triple("image_std"),
            image_to_rgb=image_to_rgb,
            backbone_depth=_integer(
                raw["backbone_depth"], where="configuration backbone_depth"
            ),
            feature_shapes=tuple(
                (
                    _integer(
                        _array(shape, where="feature shape")[0],
                        where="feature shape height",
                    ),
                    _integer(
                        _array(shape, where="feature shape")[1],
                        where="feature shape width",
                    ),
                )
                if len(_array(shape, where="feature shape")) == 2
                else (_raise_feature_shape())
                for shape in feature_shapes_raw
            ),
            embed_dim=_integer(raw["embed_dim"], where="configuration embed_dim"),
            bev_hw=pair("bev_hw"),
            encoder_layers=_integer(
                raw["encoder_layers"], where="configuration encoder_layers"
            ),
            decoder_layers=_integer(
                raw["decoder_layers"], where="configuration decoder_layers"
            ),
            object_queries=_integer(
                raw["object_queries"], where="configuration object_queries"
            ),
            class_names=tuple(
                _string(value, where="configuration class name")
                for value in class_names_raw
            ),
            attention_heads=_integer(
                raw["attention_heads"], where="configuration attention_heads"
            ),
            sampling_points=_integer(
                raw["sampling_points"], where="configuration sampling_points"
            ),
            training_queue_length=_integer(
                raw["training_queue_length"],
                where="configuration training_queue_length",
            ),
            can_bus_dim=_integer(
                raw["can_bus_dim"], where="configuration can_bus_dim"
            ),
            bbox_code_size=_integer(
                raw["bbox_code_size"], where="configuration bbox_code_size"
            ),
        )
        result.validate_tiny_profile()
        return result


def _raise_feature_shape() -> tuple[int, int]:
    raise BevFormerExportError("each feature shape must be [H, W]")


@dataclass(frozen=True, slots=True)
class ConfigExtraction:
    """Static values and provenance extracted from one config file."""

    config: BevFormerTinyStaticConfig
    source_path: str
    source_sha256: str
    audit: PythonAuditReport

    def __post_init__(self) -> None:
        if not isinstance(self.config, BevFormerTinyStaticConfig):
            raise BevFormerExportError("extraction config has the wrong type")
        _string(self.source_path, where="config source path")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.source_sha256)
        ):
            raise BevFormerExportError("config source sha256 must be lowercase hex")
        if not isinstance(self.audit, PythonAuditReport):
            raise BevFormerExportError("extraction audit has the wrong type")
        if not self.audit.safe_for_static_config:
            raise BevFormerExportError("unsafe config cannot form an extraction")


def _derive_static_config(symbols: Mapping[str, Any]) -> BevFormerTinyStaticConfig:
    model = _mapping(_at(symbols, "model"), where="model")
    head = _mapping(_at(model, "pts_bbox_head"), where="model.pts_bbox_head")
    transformer = _mapping(
        _at(head, "transformer"), where="model.pts_bbox_head.transformer"
    )
    encoder = _mapping(_at(transformer, "encoder"), where="transformer.encoder")
    decoder = _mapping(_at(transformer, "decoder"), where="transformer.decoder")
    positional_encoding = _mapping(
        _at(head, "positional_encoding"),
        where="model.pts_bbox_head.positional_encoding",
    )
    neck = _mapping(_at(model, "img_neck"), where="model.img_neck")
    backbone = _mapping(_at(model, "img_backbone"), where="model.img_backbone")
    data = _mapping(_at(symbols, "data"), where="data")
    test_pipeline = _at(symbols, "test_pipeline")

    augmentation = _one_typed(test_pipeline, "MultiScaleFlipAug3D")
    image_scale = _array(
        _at(augmentation, "img_scale"), where="MultiScaleFlipAug3D.img_scale"
    )
    if len(image_scale) != 2:
        raise BevFormerExportError("img_scale must be [width, height]")
    source_w = _integer(image_scale[0], where="img_scale width")
    source_h = _integer(image_scale[1], where="img_scale height")
    scaler = _one_typed(test_pipeline, "RandomScaleImageMultiViewImage")
    scales = _array(_at(scaler, "scales"), where="image scales")
    if len(scales) != 1:
        raise BevFormerExportError("frozen profile requires exactly one image scale")
    scale = _number(scales[0], where="image scale")
    content_h_float = source_h * scale
    content_w_float = source_w * scale
    if not content_h_float.is_integer() or not content_w_float.is_integer():
        raise BevFormerExportError("scaled image dimensions must be integral")
    content_h = int(content_h_float)
    content_w = int(content_w_float)
    padder = _one_typed(test_pipeline, "PadMultiViewImage")
    divisor = _integer(_at(padder, "size_divisor"), where="padding divisor")
    input_h = ((content_h + divisor - 1) // divisor) * divisor
    input_w = ((content_w + divisor - 1) // divisor) * divisor

    normalizer = _one_typed(test_pipeline, "NormalizeMultiviewImage")
    mean_values = _array(_at(normalizer, "mean"), where="normalization mean")
    std_values = _array(_at(normalizer, "std"), where="normalization std")
    if len(mean_values) != 3 or len(std_values) != 3:
        raise BevFormerExportError("normalization mean/std must have 3 values")
    to_rgb = _at(normalizer, "to_rgb")
    if not isinstance(to_rgb, bool):
        raise BevFormerExportError("normalization to_rgb must be boolean")

    spatial = _one_typed(encoder, "SpatialCrossAttention")
    deformable = _mapping(
        _at(spatial, "deformable_attention"),
        where="SpatialCrossAttention.deformable_attention",
    )
    decoder_mha = _one_typed(decoder, "MultiheadAttention")

    levels = _integer(_at(neck, "num_outs"), where="img_neck.num_outs")
    configured_levels = _integer(
        _at(symbols, "_num_levels_"), where="_num_levels_"
    )
    deformable_levels = _integer(
        _at(deformable, "num_levels"), where="deformable_attention.num_levels"
    )
    if not levels == configured_levels == deformable_levels:
        raise BevFormerExportError("feature level counts disagree in config")

    # The selected ResNet output is C5 (out_indices=(3,)), whose stride is 32.
    out_indices = tuple(_array(_at(backbone, "out_indices"), where="out_indices"))
    if out_indices != (3,):
        raise BevFormerExportError("frozen profile requires ResNet C5 out_indices=(3,)")
    feature_shapes = ((input_h // 32, input_w // 32),)
    if levels != 1 or input_h % 32 or input_w % 32:
        raise BevFormerExportError("frozen C5 profile requires one 32-aligned level")

    classes = tuple(
        _string(value, where="class name")
        for value in _array(_at(symbols, "class_names"), where="class_names")
    )
    configured_classes = _integer(_at(head, "num_classes"), where="num_classes")
    coder_classes = _integer(
        _at(head, "bbox_coder", "num_classes"), where="bbox_coder.num_classes"
    )
    if len(classes) != configured_classes or len(classes) != coder_classes:
        raise BevFormerExportError("class counts disagree in config")

    use_camera = _at(symbols, "input_modality", "use_camera")
    if use_camera is not True:
        raise BevFormerExportError("frozen profile requires camera input")
    batch = _integer(_at(data, "samples_per_gpu"), where="samples_per_gpu")

    config = BevFormerTinyStaticConfig(
        batch=batch,
        cameras=len(NUSCENES_CAMERA_ORDER),
        image_channels=3,
        image_content_hw=(content_h, content_w),
        image_input_hw=(input_h, input_w),
        image_mean=tuple(_number(v, where="image mean") for v in mean_values),  # type: ignore[arg-type]
        image_std=tuple(_number(v, where="image std") for v in std_values),  # type: ignore[arg-type]
        image_to_rgb=to_rgb,
        backbone_depth=_integer(_at(backbone, "depth"), where="backbone depth"),
        feature_shapes=feature_shapes,
        embed_dim=_integer(_at(symbols, "_dim_"), where="_dim_"),
        bev_hw=(
            _integer(_at(head, "bev_h"), where="bev_h"),
            _integer(_at(head, "bev_w"), where="bev_w"),
        ),
        encoder_layers=_integer(
            _at(encoder, "num_layers"), where="encoder.num_layers"
        ),
        decoder_layers=_integer(
            _at(decoder, "num_layers"), where="decoder.num_layers"
        ),
        object_queries=_integer(_at(head, "num_query"), where="num_query"),
        class_names=classes,
        attention_heads=_integer(
            _at(decoder_mha, "num_heads"), where="MultiheadAttention.num_heads"
        ),
        sampling_points=_integer(
            _at(deformable, "num_points"), where="deformable_attention.num_points"
        ),
        training_queue_length=_integer(
            _at(symbols, "queue_length"), where="queue_length"
        ),
        # These two values are fixed source-level defaults outside this config:
        # the nuScenes converter emits 18 can-bus scalars and BEVFormerHead uses
        # code_size=10 when the config does not override it.
        can_bus_dim=18,
        bbox_code_size=10,
    )
    if _integer(_at(transformer, "embed_dims"), where="transformer.embed_dims") != config.embed_dim:
        raise BevFormerExportError("transformer embed_dims disagrees with _dim_")
    positional_type = _string(
        _at(positional_encoding, "type"), where="positional_encoding.type"
    )
    if positional_type != "LearnedPositionalEncoding":
        raise BevFormerExportError(
            "frozen profile requires LearnedPositionalEncoding"
        )
    if config.embed_dim % 2:
        raise BevFormerExportError(
            "LearnedPositionalEncoding requires an even embedding dimension"
        )
    positional_num_feats = _integer(
        _at(positional_encoding, "num_feats"),
        where="positional_encoding.num_feats",
    )
    if positional_num_feats != config.embed_dim // 2:
        raise BevFormerExportError(
            "positional_encoding.num_feats must equal half the embedding dimension"
        )
    row_num_embed = _integer(
        _at(positional_encoding, "row_num_embed"),
        where="positional_encoding.row_num_embed",
    )
    col_num_embed = _integer(
        _at(positional_encoding, "col_num_embed"),
        where="positional_encoding.col_num_embed",
    )
    if row_num_embed != config.bev_hw[0]:
        raise BevFormerExportError(
            "positional_encoding.row_num_embed must equal bev_h"
        )
    if col_num_embed != config.bev_hw[1]:
        raise BevFormerExportError(
            "positional_encoding.col_num_embed must equal bev_w"
        )
    config.validate_tiny_profile()
    return config


def extract_bevformer_tiny_config(path: str | Path) -> ConfigExtraction:
    """Extract the frozen profile from ``bevformer_tiny.py`` using only AST.

    The file is never imported, compiled, or evaluated by Python.  Unsupported
    calls and control flow are reported before any values are interpreted.
    """

    config_path = Path(path)
    try:
        raw_bytes = config_path.read_bytes()
        source = raw_bytes.decode("utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise BevFormerExportError(f"cannot read {config_path}: {exc}") from exc
    source_name = config_path.as_posix()
    tree, report = _parse_and_audit(source, source_name=source_name)
    if tree is None or report.findings:
        raise BevFormerExportError(
            f"{source_name} is not a safe static config", audit_report=report
        )
    symbols, evaluation_findings = _evaluate_assignments(tree)
    if evaluation_findings:
        report = replace(report, findings=evaluation_findings)
        raise BevFormerExportError(
            f"{source_name} cannot be resolved statically", audit_report=report
        )
    config = _derive_static_config(symbols)
    return ConfigExtraction(
        config=config,
        source_path=source_name,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        audit=report,
    )


@dataclass(frozen=True, slots=True)
class TensorABI:
    """One static, contiguous tensor at the framework-independent step boundary."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    layout: str
    semantic: str

    def __post_init__(self) -> None:
        _string(self.name, where="tensor name")
        if self.dtype not in _DTYPES:
            raise BevFormerExportError(f"unsupported ABI dtype {self.dtype!r}")
        if not isinstance(self.shape, tuple) or not self.shape:
            raise BevFormerExportError("tensor shape must be a non-empty tuple")
        for dim in self.shape:
            _integer(dim, where=f"tensor {self.name} shape dimension")
        _string(self.layout, where=f"tensor {self.name} layout")
        _string(self.semantic, where=f"tensor {self.name} semantic")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "layout": self.layout,
            "semantic": self.semantic,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TensorABI:
        raw = _mapping(raw, where="tensor ABI")
        _strict_keys(
            raw,
            {"name", "dtype", "shape", "layout", "semantic"},
            where="tensor ABI",
        )
        shape_raw = _array(raw["shape"], where="tensor ABI shape")
        return cls(
            name=_string(raw["name"], where="tensor ABI name"),
            dtype=_string(raw["dtype"], where="tensor ABI dtype"),
            shape=tuple(
                _integer(value, where="tensor ABI shape dimension")
                for value in shape_raw
            ),
            layout=_string(raw["layout"], where="tensor ABI layout"),
            semantic=_string(raw["semantic"], where="tensor ABI semantic"),
        )

    def validate_array(self, value: Any) -> None:
        if not isinstance(value, np.ndarray):
            raise BevFormerExportError(f"tensor {self.name!r} must be a NumPy array")
        if value.shape != self.shape:
            raise BevFormerExportError(
                f"tensor {self.name!r} shape {value.shape} != {self.shape}"
            )
        if value.dtype != np.dtype(self.dtype):
            raise BevFormerExportError(
                f"tensor {self.name!r} dtype {value.dtype} != {self.dtype}"
            )
        if not value.flags.c_contiguous:
            raise BevFormerExportError(f"tensor {self.name!r} must be C-contiguous")
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise BevFormerExportError(f"tensor {self.name!r} contains NaN or infinity")


def _input_abi(config: BevFormerTinyStaticConfig) -> tuple[TensorABI, ...]:
    batch = config.batch
    cameras = config.cameras
    height, width = config.image_input_hw
    queries = config.bev_queries
    channels = config.embed_dim
    return (
        TensorABI(
            "images",
            "float16",
            (batch, cameras, config.image_channels, height, width),
            "NCHW-per-camera",
            "normalized-and-padded-multiview-images",
        ),
        TensorABI(
            "lidar2img",
            "float32",
            (batch, cameras, 4, 4),
            "row-major",
            "lidar-to-padded-image-projection",
        ),
        TensorABI(
            "can_bus_delta",
            "float32",
            (batch, config.can_bus_dim),
            "row-major",
            "relative-ego-motion-with-zero-delta-on-reset",
        ),
        TensorABI(
            "prev_bev",
            "float16",
            (batch, queries, channels),
            "B-HW-C",
            "previous-frame-bev-state",
        ),
        TensorABI(
            "has_prev_bev",
            "int32",
            (batch,),
            "row-major",
            "binary-state-validity-mask",
        ),
    )


def _output_abi(config: BevFormerTinyStaticConfig) -> tuple[TensorABI, ...]:
    return (
        TensorABI(
            "cls_logits",
            "float32",
            (config.batch, config.object_queries, config.num_classes),
            "B-Q-C",
            "raw-final-decoder-class-logits-before-topk",
        ),
        TensorABI(
            "bbox_regression",
            "float32",
            (config.batch, config.object_queries, config.bbox_code_size),
            "B-Q-C",
            "raw-final-decoder-box-regression-before-decode",
        ),
        TensorABI(
            "next_bev",
            "float16",
            (config.batch, config.bev_queries, config.embed_dim),
            "B-HW-C",
            "next-frame-bev-state",
        ),
    )


@dataclass(frozen=True, slots=True)
class BevFormerStepManifest:
    """Stable, JSON-serializable tensor ABI for one streaming inference step."""

    format: str
    schema_version: int
    model: str
    profile: str
    entrypoint: str
    status: str
    camera_order: tuple[str, ...]
    inputs: tuple[TensorABI, ...]
    outputs: tuple[TensorABI, ...]
    configuration: BevFormerTinyStaticConfig
    config_source: str
    config_sha256: str
    audit: PythonAuditReport
    unavailable_exports: tuple[str, ...]
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.format != MANIFEST_FORMAT:
            raise BevFormerExportError(f"unsupported manifest format {self.format!r}")
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise BevFormerExportError(
                f"unsupported manifest schema_version {self.schema_version!r}"
            )
        if self.model != "BEVFormer-tiny":
            raise BevFormerExportError("manifest model must be BEVFormer-tiny")
        if self.profile != PROFILE_NAME:
            raise BevFormerExportError("manifest profile is not the frozen tiny profile")
        if self.entrypoint != ENTRYPOINT:
            raise BevFormerExportError("manifest entrypoint is not stable")
        if self.status != "offline-contract-only":
            raise BevFormerExportError("manifest status must be offline-contract-only")
        if self.camera_order != NUSCENES_CAMERA_ORDER:
            raise BevFormerExportError("manifest camera order is not the frozen order")
        if not isinstance(self.configuration, BevFormerTinyStaticConfig):
            raise BevFormerExportError("manifest configuration has the wrong type")
        self.configuration.validate_tiny_profile()
        if self.inputs != _input_abi(self.configuration):
            raise BevFormerExportError("manifest input ABI does not match configuration")
        if self.outputs != _output_abi(self.configuration):
            raise BevFormerExportError("manifest output ABI does not match configuration")
        input_names = tuple(item.name for item in self.inputs)
        output_names = tuple(item.name for item in self.outputs)
        if input_names != (
            "images",
            "lidar2img",
            "can_bus_delta",
            "prev_bev",
            "has_prev_bev",
        ):
            raise BevFormerExportError("manifest input order is not stable")
        if output_names != ("cls_logits", "bbox_regression", "next_bev"):
            raise BevFormerExportError("manifest output order is not stable")
        _string(self.config_source, where="manifest config_source")
        if (
            not isinstance(self.config_sha256, str)
            or len(self.config_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.config_sha256)
        ):
            raise BevFormerExportError("manifest config_sha256 must be lowercase hex")
        if not isinstance(self.audit, PythonAuditReport) or not self.audit.safe_for_static_config:
            raise BevFormerExportError("manifest needs a successful static audit")
        if not isinstance(self.unavailable_exports, tuple) or set(
            self.unavailable_exports
        ) != {"torch.export", "onnx"}:
            raise BevFormerExportError(
                "manifest must state that torch.export and ONNX are unavailable"
            )
        if not isinstance(self.assumptions, tuple) or not self.assumptions or not all(
            isinstance(item, str) and item for item in self.assumptions
        ):
            raise BevFormerExportError("manifest assumptions must be non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "model": self.model,
            "profile": self.profile,
            "entrypoint": self.entrypoint,
            "status": self.status,
            "camera_order": list(self.camera_order),
            "inputs": [item.to_dict() for item in self.inputs],
            "outputs": [item.to_dict() for item in self.outputs],
            "configuration": self.configuration.to_dict(),
            "config_source": self.config_source,
            "config_sha256": self.config_sha256,
            "audit": self.audit.to_dict(),
            "unavailable_exports": list(self.unavailable_exports),
            "assumptions": list(self.assumptions),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> BevFormerStepManifest:
        raw = _mapping(raw, where="manifest")
        keys = {
            "format",
            "schema_version",
            "model",
            "profile",
            "entrypoint",
            "status",
            "camera_order",
            "inputs",
            "outputs",
            "configuration",
            "config_source",
            "config_sha256",
            "audit",
            "unavailable_exports",
            "assumptions",
        }
        _strict_keys(raw, keys, where="manifest")
        camera_order = _array(raw["camera_order"], where="camera_order")
        inputs = _array(raw["inputs"], where="inputs")
        outputs = _array(raw["outputs"], where="outputs")
        unavailable = _array(
            raw["unavailable_exports"], where="unavailable_exports"
        )
        assumptions = _array(raw["assumptions"], where="assumptions")
        return cls(
            format=_string(raw["format"], where="manifest format"),
            schema_version=_integer(
                raw["schema_version"], where="manifest schema_version"
            ),
            model=_string(raw["model"], where="manifest model"),
            profile=_string(raw["profile"], where="manifest profile"),
            entrypoint=_string(raw["entrypoint"], where="manifest entrypoint"),
            status=_string(raw["status"], where="manifest status"),
            camera_order=tuple(
                _string(item, where="camera name") for item in camera_order
            ),
            inputs=tuple(
                TensorABI.from_dict(_mapping(item, where="input ABI"))
                for item in inputs
            ),
            outputs=tuple(
                TensorABI.from_dict(_mapping(item, where="output ABI"))
                for item in outputs
            ),
            configuration=BevFormerTinyStaticConfig.from_dict(
                _mapping(raw["configuration"], where="configuration")
            ),
            config_source=_string(
                raw["config_source"], where="manifest config_source"
            ),
            config_sha256=_string(
                raw["config_sha256"], where="manifest config_sha256"
            ),
            audit=PythonAuditReport.from_dict(
                _mapping(raw["audit"], where="audit report")
            ),
            unavailable_exports=tuple(
                _string(item, where="unavailable export") for item in unavailable
            ),
            assumptions=tuple(
                _string(item, where="manifest assumption") for item in assumptions
            ),
        )

    @classmethod
    def from_json(cls, text: str) -> BevFormerStepManifest:
        if not isinstance(text, str):
            raise BevFormerExportError("manifest JSON must be text")

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise BevFormerExportError(f"duplicate JSON key {key!r}")
                result[key] = value
            return result

        def reject_constant(value: str) -> None:
            raise BevFormerExportError(f"invalid JSON numeric constant {value}")

        try:
            raw = json.loads(
                text,
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_constant,
            )
        except json.JSONDecodeError as exc:
            raise BevFormerExportError(f"invalid manifest JSON: {exc}") from exc
        return cls.from_dict(_mapping(raw, where="manifest"))

    def validate_inputs(self, values: Mapping[str, Any]) -> None:
        self._validate_tensors(values, self.inputs, kind="input")
        state_flag = values["has_prev_bev"]
        if not np.isin(state_flag, np.asarray([0, 1], dtype=np.int32)).all():
            raise BevFormerExportError("has_prev_bev accepts only 0 or 1")
        reset_rows = state_flag == 0
        if np.any(reset_rows):
            reset_can_bus = values["can_bus_delta"][reset_rows]
            translation_is_zero = np.equal(
                reset_can_bus[:, :3], np.float32(0.0)
            ).all()
            rotation_is_zero = np.equal(
                reset_can_bus[:, -1], np.float32(0.0)
            ).all()
            if not translation_is_zero or not rotation_is_zero:
                raise BevFormerExportError(
                    "can_bus_delta translation [:3] and frame rotation [-1] "
                    "must be zero when has_prev_bev is zero"
                )

    def validate_outputs(self, values: Mapping[str, Any]) -> None:
        self._validate_tensors(values, self.outputs, kind="output")

    @staticmethod
    def _validate_tensors(
        values: Mapping[str, Any], specs: tuple[TensorABI, ...], *, kind: str
    ) -> None:
        if not isinstance(values, Mapping) or not all(
            isinstance(name, str) for name in values
        ):
            raise BevFormerExportError(f"{kind}s must be a string-keyed mapping")
        expected = {item.name for item in specs}
        actual = set(values)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise BevFormerExportError(
                f"{kind} tensor names differ: missing={missing}, extra={extra}"
            )
        for spec in specs:
            spec.validate_array(values[spec.name])


def build_bevformer_tiny_manifest(
    config_path: str | Path, *, source_label: str | None = None
) -> BevFormerStepManifest:
    """Create the offline step manifest from an audited local config."""

    extraction = extract_bevformer_tiny_config(config_path)
    label = source_label if source_label is not None else extraction.source_path
    label = _string(label, where="source label")
    audit = replace(extraction.audit, source=label)
    return BevFormerStepManifest(
        format=MANIFEST_FORMAT,
        schema_version=MANIFEST_SCHEMA_VERSION,
        model="BEVFormer-tiny",
        profile=PROFILE_NAME,
        entrypoint=ENTRYPOINT,
        status="offline-contract-only",
        camera_order=NUSCENES_CAMERA_ORDER,
        inputs=_input_abi(extraction.config),
        outputs=_output_abi(extraction.config),
        configuration=extraction.config,
        config_source=label,
        config_sha256=extraction.source_sha256,
        audit=audit,
        unavailable_exports=("torch.export", "onnx"),
        assumptions=(
            "camera order comes from tools/data_converter/nuscenes_converter.py",
            "can_bus_dim=18 comes from the nuScenes converter output",
            "bbox_code_size=10 comes from BEVFormerHead's default",
            "input preprocessing is outside the compiled step",
            "this artifact defines an offline ABI and does not export a framework model",
        ),
    )


def load_bevformer_manifest(path: str | Path) -> BevFormerStepManifest:
    manifest_path = Path(path)
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BevFormerExportError(f"cannot read {manifest_path}: {exc}") from exc
    return BevFormerStepManifest.from_json(text)


def save_bevformer_manifest(
    manifest: BevFormerStepManifest, path: str | Path
) -> None:
    if not isinstance(manifest, BevFormerStepManifest):
        raise BevFormerExportError("manifest has the wrong type")
    manifest.validate()
    manifest_path = Path(path)
    manifest_path.write_text(manifest.to_json() + "\n", encoding="utf-8")


# Short aliases for callers that already selected the BEVFormer frontend.
build_manifest = build_bevformer_tiny_manifest
load_manifest = load_bevformer_manifest
save_manifest = save_bevformer_manifest


__all__ = [
    "AuditFinding",
    "BevFormerExportError",
    "BevFormerStepManifest",
    "BevFormerTinyStaticConfig",
    "ConfigExtraction",
    "ENTRYPOINT",
    "MANIFEST_FORMAT",
    "MANIFEST_SCHEMA_VERSION",
    "NUSCENES_CAMERA_ORDER",
    "PROFILE_NAME",
    "PythonAuditReport",
    "TensorABI",
    "audit_python_file",
    "audit_python_source",
    "build_bevformer_tiny_manifest",
    "build_manifest",
    "extract_bevformer_tiny_config",
    "load_bevformer_manifest",
    "load_manifest",
    "save_bevformer_manifest",
    "save_manifest",
]
