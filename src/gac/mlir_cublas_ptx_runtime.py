"""Runtime for cuBLAS MatMul calls and MLIR-generated fusion PTX."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
from threading import Lock
import time
from typing import Any

from .backends.mlir_cublas_ptx import (
    COMPUTE_CAPABILITY,
    MlirCublasPtxProgram,
    validate_mlir_cublas_ptx_program,
)
from .backends.mlir_cuda import MlirCudaCall
from .errors import BackendError, DeviceUnavailableError, ValidationError
from .mlir_runtime import MlirCudaSession
from .runtime import _cupy, cuda_device_info


class MlirCublasPtxSession(MlirCudaSession):
    """Execute two cuBLAS calls and two precompiled PTX fusion kernels.

    Artifact validation is completed before CUDA discovery.  The PTX is then
    loaded by path through the CUDA Driver; this runtime never submits CUDA C
    to NVRTC and has no generated-CUDA fallback.
    """

    def __init__(
        self, program: MlirCublasPtxProgram, *, device_id: int = 0
    ) -> None:
        if not isinstance(program, MlirCublasPtxProgram):
            raise ValidationError("program must be an MlirCublasPtxProgram")
        validate_mlir_cublas_ptx_program(program)

        started = time.perf_counter()
        self.program = program
        self._graph = program.graph.clone()
        self.info = cuda_device_info(device_id)
        if tuple(self.info["compute_capability"]) != COMPUTE_CAPABILITY:
            raise DeviceUnavailableError(
                "mlir-cublas-ptx-sm86 requires compute capability 8.6, got "
                f"{self.info['compute_capability']}"
            )
        self.info["compiler_backend"] = "mlir-cublas-ptx-sm86"
        self.info["matmul_backend"] = {
            "selected": "cupy-cublas",
            "call_count": sum(
                call.implementation == "cupy-cublas" for call in program.calls
            ),
            "precision": "float32",
        }
        self.info["fusion_backend"] = {
            "selected": "official-mlir-nvvm-ptx",
            "launch_count": sum(
                call.implementation == "mlir-ptx" for call in program.calls
            ),
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
        self._module: Any | None = None
        self._stream: Any | None = None
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._ptx_path: Path | None = None

        try:
            with self._device:
                self._stream = self._cp.cuda.Stream(non_blocking=True)
                with self._stream:
                    self._temporary_directory = tempfile.TemporaryDirectory(
                        prefix="gac-mlir-cublas-ptx-"
                    )
                    self._ptx_path = (
                        Path(self._temporary_directory.name) / "fusions.ptx"
                    )
                    self._ptx_path.write_text(
                        program.ptx, encoding="utf-8", newline="\n"
                    )
                    self._module = self._cp.RawModule(path=str(self._ptx_path))
                    for call in program.calls:
                        if call.implementation == "mlir-ptx":
                            self._kernels[call.symbol] = self._module.get_function(
                                call.symbol
                            )
                        elif call.implementation != "cupy-cublas":
                            raise BackendError(
                                "unsupported MLIR cuBLAS/PTX implementation "
                                f"{call.implementation!r}"
                            )

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
            raise BackendError(
                f"MLIR cuBLAS/PTX session initialization failed: {exc}"
            ) from exc

        self.setup_ms = (time.perf_counter() - started) * 1000.0

    def _clear(self) -> None:
        super()._clear()
        self._ptx_path = None
        temporary_directory = self._temporary_directory
        self._temporary_directory = None
        if temporary_directory is not None:
            temporary_directory.cleanup()

    def _validate_call_abi(self, call: MlirCudaCall) -> None:
        if call.implementation == "mlir-ptx":
            # The tensor, launch, aliasing and provenance ABI is intentionally
            # identical to the established fused CUDA call plan.  Only the
            # implementation changes from NVRTC CUDA to precompiled PTX.
            call = replace(call, implementation="nvrtc-cuda")
        super()._validate_call_abi(call)

    def _validate_call_plan(self) -> None:
        validate_mlir_cublas_ptx_program(self.program)
        implementations = tuple(call.implementation for call in self.program.calls)
        if implementations != (
            "cupy-cublas",
            "mlir-ptx",
            "cupy-cublas",
            "mlir-ptx",
        ):
            raise BackendError(
                "MLIR cuBLAS/PTX runtime call implementation order is invalid"
            )
        for call in self.program.calls:
            self._validate_call_abi(call)
            missing = [
                name
                for name in (*call.inputs, call.output)
                if name not in self._values
            ]
            if missing:
                raise BackendError(
                    f"MLIR cuBLAS/PTX call {call.symbol!r} references "
                    f"missing values {missing}"
                )
            if call.implementation == "mlir-ptx" and (
                call.grid is None
                or call.block is None
                or call.symbol not in self._kernels
            ):
                raise BackendError(
                    f"MLIR PTX call {call.symbol!r} has no complete launch configuration"
                )
        self._validate_actual_values()


__all__ = ["MlirCublasPtxSession"]
