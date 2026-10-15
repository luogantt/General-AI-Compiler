"""The shared ten-stage compiler contract and its ordered execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, TypeVar


@dataclass(frozen=True, slots=True)
class CompilerStage:
    """One compiler boundary and its inspectable output artifact."""

    number: int
    name: str
    contract: str
    artifact: str

    def to_dict(self) -> dict[str, int | str]:
        return {"stage": self.number, "name": self.name,
                "contract": self.contract, "artifact": self.artifact}


PIPELINE_NAME = "ten-stage compiler"
COMPILER_STAGES = (
    CompilerStage(1, "model-import", "Model / weights / ABI", "01-import.json"),
    CompilerStage(2, "ir-conversion", "Typed Graph IR, axes and shapes", "02-graph.json"),
    CompilerStage(3, "graph-optimization", "Graph rewrites and pass reports", "03-passes.json"),
    CompilerStage(4, "legalization", "Target operation, shape and precision constraints", "04-capabilities.json"),
    CompilerStage(5, "subgraph-partition", "Executable regions and data dependencies", "05-dispatch-dag.json"),
    CompilerStage(6, "template-selection", "Target compute template for each region", "06-templates.json"),
    CompilerStage(7, "tiling", "Tile shapes, reduction axes and staging requirements", "07-tiling.json"),
    CompilerStage(8, "schedule", "Dispatch order and buffer lifetimes", "08-schedule.json"),
    CompilerStage(9, "codegen", "Lower target operations to executable device code", "kernel.mlir"),
    CompilerStage(10, "runtime", "Resident data, dispatch ABI and host execution", "10-runtime.json"),
)


def pipeline_manifest() -> list[dict[str, int | str]]:
    """Return fresh records so one compilation cannot change the catalog."""
    return [stage.to_dict() for stage in COMPILER_STAGES]


State = TypeVar("State")


def run_pipeline(state: State, handlers: Mapping[str, Callable[[State], State]]) -> State:
    """Execute each target's ten stage implementations in catalog order.

    Validate the complete handler set before running anything, so a missing
    layer cannot silently produce a partially compiled program.
    """
    expected = {stage.name for stage in COMPILER_STAGES}
    actual = set(handlers)
    if actual != expected:
        raise ValueError(f"compiler stage handlers mismatch: missing={sorted(expected - actual)}, "
                         f"unexpected={sorted(actual - expected)}")
    for stage in COMPILER_STAGES:
        if not callable(handlers[stage.name]):
            raise TypeError(f"compiler stage {stage.name!r} needs a callable handler")
    for stage in COMPILER_STAGES:
        state = handlers[stage.name](state)
    return state
