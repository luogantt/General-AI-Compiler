"""CUDA runtime for the three-dispatch shared-memory tiled MLIR FFN."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
from threading import Lock
import time
from typing import Any

from gac.targets.cuda.cublas_ptx import COMPUTE_CAPABILITY
from gac.core.errors import BackendError, DeviceUnavailableError, ValidationError
from gac.execution.mlir_cuda import MlirCudaSession
from gac.targets.cuda.tiled_ffn import (
    IMPLEMENTATION,
    TARGET,
    MlirTiledFfnProgram,
    validate_mlir_tiled_ffn_program,
)
from gac.execution.cuda import _cupy, cuda_device_info


class MlirTiledFfnSession(MlirCudaSession):
    """Execute FC1+Bias+ReLU, FC2 and final LayerNorm as three PTX kernels."""

    def __init__(
        self, program: MlirTiledFfnProgram, *, device_id: int = 0
    ) -> None:
        if not isinstance(program, MlirTiledFfnProgram):
            raise ValidationError("program must be an MlirTiledFfnProgram")
        # Integrity and the tile/block-to-code relationship are checked before
        # CUDA discovery, CuPy import, PTX loading, or device allocation.
        validate_mlir_tiled_ffn_program(program)

        started = time.perf_counter()
        self.program = program
        self._graph = program.graph.clone()
        self.info = cuda_device_info(device_id)
        if tuple(self.info["compute_capability"]) != COMPUTE_CAPABILITY:
            raise DeviceUnavailableError(
                f"{TARGET} requires compute capability 8.6, got "
                f"{self.info['compute_capability']}"
            )
        self.info["compiler_backend"] = TARGET
        self.info["matmul_backend"] = {
            "selected": "mlir-shared-memory-tiled-ptx",
            "call_count": 2,
            "precision": "float32",
            "tile": list(program.matmul_tile),
            "thread_block": list(program.thread_block),
            "single_buffered": True,
        }
        self.info["fusion_backend"] = {
            "selected": "official-mlir-nvvm-ptx",
            "launch_count": 2,
            "dispatch_count": 3,
            "fused_fc1_bias_relu": True,
            "fused_final_bias_residual_layernorm": True,
            "nvrtc_launch_count": 0,
        }

        self._cp = _cupy()
        self._device = self._cp.cuda.Device(device_id)
        self._lock = Lock()
        self._closed = False
        self._failed = False
        self._constants: dict[str, Any] = {}
        self._inputs: dict[str, Any] = {}
        self._outputs: dict[str, Any] = {}
        self._workspaces: dict[str, Any] = {}
        self._values: dict[str, Any] = {}
        self._kernels: dict[str, Any] = {}
        self._modules: dict[str, Any] = {}
        self._module: Any | None = None
        self._stream: Any | None = None
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._ptx_paths: dict[str, Path] = {}

        try:
            with self._device:
                self._stream = self._cp.cuda.Stream(non_blocking=True)
                with self._stream:
                    self._temporary_directory = tempfile.TemporaryDirectory(
                        prefix="gac-mlir-tiled-ffn-"
                    )
                    root = Path(self._temporary_directory.name)
                    module_specs = (
                        ("fc1", program.fc1.ptx, program.calls[0].symbol),
                        ("fc2", program.fc2.ptx, program.calls[1].symbol),
                        ("final", program.base_program.ptx, program.calls[2].symbol),
                    )
                    for component, ptx, symbol in module_specs:
                        path = root / f"{component}.ptx"
                        path.write_text(ptx, encoding="utf-8", newline="\n")
                        module = self._cp.RawModule(path=str(path))
                        self._ptx_paths[component] = path
                        self._modules[component] = module
                        self._kernels[symbol] = module.get_function(symbol)

                    for name, value in self._graph.constants.items():
                        dtype = self._graph.value_specs[name].dtype
                        self._constants[name] = self._cp.array(
                            value, dtype=dtype, order="C", copy=True
                        )
                    for name, spec in self._graph.inputs.items():
                        self._inputs[name] = self._cp.empty(
                            spec.shape, dtype=spec.dtype, order="C"
                        )
                    for name in self._graph.outputs:
                        spec = self._graph.value_specs[name]
                        self._outputs[name] = self._cp.empty(
                            spec.shape, dtype=spec.dtype, order="C"
                        )
                    self._values = dict(self._constants)
                    self._values.update(self._inputs)
                    self._allocate_call_values()
                    self._validate_call_plan()
                    self._stream.synchronize()
                    self.kernel_info = [
                        self._call_info(call) for call in program.calls
                    ]
        except BaseException as exc:
            self._closed = True
            try:
                if self._stream is not None:
                    with self._device:
                        self._stream.synchronize()
            except Exception:
                pass
            finally:
                self._clear()
            if isinstance(exc, (BackendError, DeviceUnavailableError, ValidationError)):
                raise
            if not isinstance(exc, Exception):
                raise
            raise BackendError(f"MLIR tiled FFN session initialization failed: {exc}") from exc

        self.setup_ms = (time.perf_counter() - started) * 1000.0

    def _clear(self) -> None:
        super()._clear()
        self._modules.clear()
        self._ptx_paths.clear()
        temporary_directory = self._temporary_directory
        self._temporary_directory = None
        if temporary_directory is not None:
            temporary_directory.cleanup()

    def _validate_call_abi(self, call) -> None:
        if call == self.program.calls[2]:
            super()._validate_call_abi(
                replace(call, implementation="nvrtc-cuda")
            )
            return
        if call not in self.program.calls[:2]:
            raise BackendError(f"unknown MLIR tiled FFN call {call.symbol!r}")
        expected = self.program.calls[0 if call is self.program.calls[0] else 1]
        if call != expected or call.implementation != IMPLEMENTATION:
            raise BackendError(
                f"MLIR tiled FFN call {call.symbol!r} differs from its compiled artifact"
            )
        config = self.program.fc1.config if call is self.program.calls[0] else self.program.fc2.config
        expected_inputs = 3 if config.fuse_bias_relu else 2
        if (
            call.kind not in {"matmul", "fused-matmul"}
            or len(call.inputs) != expected_inputs
            or call.output in call.inputs
            or call.grid != config.grid
            or call.block != config.block
            or (call.m, call.n, call.k) != (config.m, config.n, config.k)
            or call.dtype != "float32"
            or call.layout != "contiguous"
            or call.aliases_input is not None
            or call.transpose_a is not False
            or call.transpose_b is not False
            or call.alpha != 1.0
            or call.beta != 0.0
        ):
            raise BackendError(f"MLIR tiled FFN call {call.symbol!r} has an invalid ABI")

    def _validate_call_plan(self) -> None:
        validate_mlir_tiled_ffn_program(self.program)
        if len(self.program.calls) != 3:
            raise BackendError("MLIR tiled FFN requires exactly three dispatches")
        for call in self.program.calls:
            self._validate_call_abi(call)
            missing = [
                name for name in (*call.inputs, call.output) if name not in self._values
            ]
            if missing:
                raise BackendError(
                    f"MLIR tiled FFN call {call.symbol!r} references missing values {missing}"
                )
            if call.grid is None or call.block is None or call.symbol not in self._kernels:
                raise BackendError(
                    f"MLIR tiled FFN call {call.symbol!r} has no complete PTX launch"
                )
        self._validate_actual_values()

    def _execute_call(self, call) -> None:
        values = self._values
        if call == self.program.calls[0]:
            arguments = tuple(values[name] for name in call.inputs) + (
                values[call.output],
            )
        elif call == self.program.calls[1]:
            arguments = tuple(values[name] for name in call.inputs) + (
                values[call.output],
            )
        elif call == self.program.calls[2]:
            arguments = tuple(values[name] for name in call.inputs) + (
                values[call.output],
            )
        else:
            raise BackendError(f"unknown MLIR tiled FFN symbol {call.symbol!r}")
        assert call.grid is not None and call.block is not None
        self._kernels[call.symbol](
            call.grid,
            call.block,
            arguments,
            stream=self._stream,
        )


__all__ = ["MlirTiledFfnSession"]
