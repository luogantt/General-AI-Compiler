"""Execute the *same* scalar MLIR bodies on the host for numerical testing.

This is an explicit compiler validation harness, never an inference fallback.
It does not exercise Vortex CSRs, SIMT reconvergence or the device Runtime.
Those require a separate SimX/RTL/FPGA execution of ``kernel.mlir``.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess
import tempfile
from typing import Mapping

import numpy as np

from ..errors import BackendError
from ..mlir_toolchain import verify_mlir
from ..runtime import bind_inputs
from .vortex_graph import LOWERING_PASSES, VortexGraphProgram


class NativeVortexGraphSession:
    """Compile SCF/Arith/LLVM bodies to host machine code and execute the DAG."""

    def __init__(self, program: VortexGraphProgram, *, llvm_bin: str | Path,
                 timeout: float = 180.0):
        self.program = program
        self.directory = tempfile.TemporaryDirectory(prefix="gac-vortex-native-")
        root = Path(self.directory.name)
        tools = Path(llvm_bin)
        source = verify_mlir(program.native_source, command=(str(tools / "mlir-opt"), *LOWERING_PASSES), timeout=timeout)
        (root / "kernel.mlir").write_text(source, encoding="utf-8")
        for command in (
            [str(tools / "mlir-translate"), "--mlir-to-llvmir", str(root / "kernel.mlir"), "-o", str(root / "kernel.ll")],
            [str(tools / "clang"), "-O2", "-shared", "-fPIC", str(root / "kernel.ll"), "-lm", "-o", str(root / "kernel.so")],
        ):
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
            if result.returncode:
                raise BackendError("native MLIR validation failed: " + result.stderr[-6000:])
        self.library = ctypes.CDLL(str(root / "kernel.so"))
        self.functions = []
        for dispatch in program.dispatches:
            function = getattr(self.library, dispatch["kernel"] + "_native")
            function.argtypes = [ctypes.c_void_p] * (len(dispatch["inputs"]) + 1)
            function.restype = None
            self.functions.append(function)

    def run(self, inputs: Mapping[str, np.ndarray], *, capture_intermediates: bool = False) -> dict[str, np.ndarray]:
        graph, memory = self.program.graph, self.program.memory_plan
        values = {name: np.array(value, dtype=graph.inputs[name].dtype, copy=True, order="C")
                  for name, value in bind_inputs(graph, inputs).items()}
        # ascontiguousarray promotes a scalar shape () to (1,), which changes
        # the public ABI when a constant is also a graph output. np.array keeps
        # rank zero and materializes the graph's declared native storage dtype.
        values.update({name: np.array(value, dtype=graph.value_specs[name].dtype, copy=True, order="C")
                       for name, value in graph.constants.items()})
        arena = np.empty(max(memory.arena_size, 1), dtype=np.uint8)
        for allocation in memory.allocations:
            if allocation.name in values:
                continue
            spec = allocation.spec
            if allocation.storage == "arena":
                values[allocation.name] = np.ndarray(spec.shape, dtype=spec.dtype, buffer=arena, offset=allocation.offset)
            else:
                values[allocation.name] = np.empty(spec.shape, dtype=spec.dtype)
        captured = {}
        for function, dispatch in zip(self.functions, self.program.dispatches, strict=True):
            function(*(ctypes.c_void_p(values[name].ctypes.data) for name in (*dispatch["inputs"], dispatch["output"])))
            if capture_intermediates:
                captured[dispatch["output"]] = values[dispatch["output"]].copy()
        if capture_intermediates:
            captured.update({name: values[name].copy() for name in graph.outputs})
            return captured
        return {name: values[name].copy() for name in graph.outputs}

    def close(self) -> None:
        self.directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
