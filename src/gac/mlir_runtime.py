"""CUDA runtime for explicit MLIR cuBLAS and fused-kernel call plans."""

from __future__ import annotations

import math
from threading import Lock
import time
from typing import Any, Mapping

import numpy as np

from .backends.mlir_cuda import (
    COMPUTE_CAPABILITY,
    CUDA_GRID_X_MAX,
    CUDA_KERNEL_INDEX_MAX,
    MlirCudaCall,
    MlirCudaProgram,
    NVRTC_OPTIONS,
    RUNTIME_ABI_VERSION,
    validate_mlir_cuda_program,
)
from .errors import BackendError, DeviceUnavailableError, ValidationError
from .profiler import Profiler
from .runtime import RunResult, _cupy, bind_inputs, cuda_device_info


class MlirCudaSession:
    """Execute a compiled MLIR runtime-call plan on one SM86 CUDA device.

    Dispatch is driven exclusively by :class:`MlirCudaCall` records created by
    the compiler.  The runtime never guesses an implementation from graph op
    names and never falls back to NumPy or the older generated-CUDA backend.
    """

    def __init__(self, program: MlirCudaProgram, *, device_id: int = 0) -> None:
        if not isinstance(program, MlirCudaProgram):
            raise ValidationError("program must be an MlirCudaProgram")
        # This CPU-only check must precede device discovery, CuPy import,
        # NVRTC compilation and allocation.  A caller cannot execute a
        # modified source, call plan, kernel, graph, or ABI record.
        validate_mlir_cuda_program(program)
        started = time.perf_counter()
        self.program = program
        self._graph = program.graph.clone()
        self.info = cuda_device_info(device_id)
        if tuple(self.info["compute_capability"]) != COMPUTE_CAPABILITY:
            raise DeviceUnavailableError(
                "mlir-cuda-sm86 requires compute capability 8.6, got "
                f"{self.info['compute_capability']}"
            )
        self.info["compiler_backend"] = "mlir-cuda-sm86"
        self.info["matmul_backend"] = {
            "selected": "cupy-cublas",
            "call_count": sum(
                call.implementation == "cupy-cublas" for call in program.calls
            ),
            "precision": "float32",
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

        try:
            with self._device:
                self._stream = self._cp.cuda.Stream(non_blocking=True)
                with self._stream:
                    self._module = self._cp.RawModule(
                        code=program.kernel_source,
                        options=NVRTC_OPTIONS,
                        backend="nvrtc",
                    )
                    self._module.compile()
                    for call in program.calls:
                        if call.implementation == "nvrtc-cuda":
                            self._kernels[call.symbol] = self._module.get_function(
                                call.symbol
                            )
                        elif call.implementation != "cupy-cublas":
                            raise BackendError(
                                f"unsupported MLIR runtime implementation {call.implementation!r}"
                            )

                    for name, value in self._graph.constants.items():
                        dtype = self._graph.value_specs[name].dtype
                        self._constants[name] = self._cp.array(
                            value, dtype=dtype, order="C", copy=True
                        )
                    for name, spec in self._graph.inputs.items():
                        self._inputs[name] = self._cp.empty(
                            spec.shape, dtype=spec.dtype
                        )
                    for name in self._graph.outputs:
                        spec = self._graph.value_specs[name]
                        self._outputs[name] = self._cp.empty(
                            spec.shape, dtype=spec.dtype
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
            raise BackendError(f"MLIR CUDA session initialization failed: {exc}") from exc

        self.setup_ms = (time.perf_counter() - started) * 1000.0

    def _clear(self) -> None:
        self._values.clear()
        self._workspaces.clear()
        self._outputs.clear()
        self._inputs.clear()
        self._constants.clear()
        self._kernels.clear()
        self._module = None
        self._stream = None

    def _allocate_call_values(self) -> None:
        """Allocate and alias values from the explicit call/dataflow records."""

        graph = self._graph
        for call in self.program.calls:
            missing = [name for name in call.inputs if name not in self._values]
            if missing:
                raise BackendError(
                    f"MLIR runtime call {call.symbol!r} uses values before definition: "
                    f"{missing}"
                )
            if call.output in self._values:
                raise BackendError(
                    f"MLIR runtime call {call.symbol!r} redefines {call.output!r}"
                )
            spec = graph.value_specs.get(call.output)
            if spec is None:
                raise BackendError(
                    f"MLIR runtime call {call.symbol!r} has no output tensor spec"
                )
            if call.aliases_input is not None:
                if (
                    isinstance(call.aliases_input, bool)
                    or not 0 <= call.aliases_input < len(call.inputs)
                ):
                    raise BackendError(
                        f"MLIR runtime call {call.symbol!r} has an invalid alias index"
                    )
                value = self._values[call.inputs[call.aliases_input]]
            elif call.output in self._outputs:
                value = self._outputs[call.output]
            else:
                value = self._cp.empty(spec.shape, dtype=spec.dtype, order="C")

            self._values[call.output] = value
            if call.output not in self._outputs:
                self._workspaces[call.output] = value

        missing_outputs = [
            name for name in graph.outputs if self._values.get(name) is not self._outputs[name]
        ]
        if missing_outputs:
            raise BackendError(
                f"MLIR runtime call plan does not define graph outputs {missing_outputs}"
            )

    @staticmethod
    def _device_pointer(value: Any) -> int | None:
        try:
            return int(value.data.ptr)
        except (AttributeError, TypeError, ValueError):
            return None

    @staticmethod
    def _is_c_contiguous(value: Any) -> bool:
        try:
            return bool(value.flags.c_contiguous)
        except AttributeError:
            try:
                return bool(value.flags["C_CONTIGUOUS"])
            except (AttributeError, KeyError, TypeError):
                return False

    def _validate_actual_values(self) -> None:
        graph = self._graph
        referenced = {
            name
            for call in self.program.calls
            for name in (*call.inputs, call.output)
        }
        for name in referenced:
            if name not in self._values or name not in graph.value_specs:
                raise BackendError(f"MLIR runtime value {name!r} is not bound")
            value = self._values[name]
            spec = graph.value_specs[name]
            try:
                shape = tuple(int(dim) for dim in value.shape)
                dtype = np.dtype(value.dtype).name
            except (AttributeError, TypeError, ValueError) as exc:
                raise BackendError(
                    f"MLIR runtime value {name!r} has no tensor ABI"
                ) from exc
            if shape != spec.shape or dtype != spec.dtype:
                raise BackendError(
                    f"MLIR runtime value {name!r} requires {spec.shape}/{spec.dtype}, "
                    f"got {shape}/{dtype}"
                )
            if not self._is_c_contiguous(value):
                raise BackendError(f"MLIR runtime value {name!r} is not C-contiguous")

        for call in self.program.calls:
            output = self._values[call.output]
            output_pointer = self._device_pointer(output)
            if call.aliases_input is not None:
                source = self._values[call.inputs[call.aliases_input]]
                if output is not source:
                    raise BackendError(
                        f"MLIR runtime call {call.symbol!r} does not satisfy object alias identity"
                    )
                source_pointer = self._device_pointer(source)
                if (
                    output_pointer is not None
                    and source_pointer is not None
                    and output_pointer != source_pointer
                ):
                    raise BackendError(
                        f"MLIR runtime call {call.symbol!r} does not satisfy pointer alias identity"
                    )
            else:
                for input_name in call.inputs:
                    input_value = self._values[input_name]
                    if output is input_value:
                        raise BackendError(
                            f"MLIR runtime call {call.symbol!r} has an undeclared object alias"
                        )
                    input_pointer = self._device_pointer(input_value)
                    if (
                        output_pointer is not None
                        and input_pointer is not None
                        and output_pointer == input_pointer
                    ):
                        raise BackendError(
                            f"MLIR runtime call {call.symbol!r} has an undeclared pointer alias"
                        )

    @staticmethod
    def _require_absent_parameters(
        call: MlirCudaCall, names: tuple[str, ...]
    ) -> None:
        present = [name for name in names if getattr(call, name) is not None]
        if present:
            raise BackendError(
                f"MLIR runtime call {call.symbol!r} has inapplicable parameters {present}"
            )

    def _validate_call_abi(self, call: MlirCudaCall) -> None:
        graph = self.program.graph
        if call.abi_version != RUNTIME_ABI_VERSION:
            raise BackendError(
                f"MLIR runtime call {call.symbol!r} has ABI version {call.abi_version}"
            )
        if call.dtype != "float32" or call.layout != "contiguous":
            raise BackendError(
                f"MLIR runtime call {call.symbol!r} requires float32/contiguous ABI"
            )
        names = (*call.inputs, call.output)
        missing_specs = [name for name in names if name not in graph.value_specs]
        if missing_specs:
            raise BackendError(
                f"MLIR runtime call {call.symbol!r} references unspecified values {missing_specs}"
            )
        for name in names:
            spec = graph.value_specs[name]
            if (
                spec.dtype != call.dtype
                or spec.layout != call.layout
                or not spec.is_static
                or any(int(dim) <= 0 for dim in spec.shape)
            ):
                raise BackendError(
                    f"MLIR runtime value {name!r} violates the call tensor ABI"
                )

        nodes = {node.id: node for node in graph.nodes}
        if not call.node_ids or any(node_id not in nodes for node_id in call.node_ids):
            raise BackendError(
                f"MLIR runtime call {call.symbol!r} has invalid node provenance"
            )
        expected_sources: list[str] = []
        for node_id in call.node_ids:
            for source in nodes[node_id].source_nodes:
                if source not in expected_sources:
                    expected_sources.append(source)
        if call.source_nodes != tuple(expected_sources):
            raise BackendError(
                f"MLIR runtime call {call.symbol!r} has invalid source provenance"
            )

        if call.symbol in {
            "gac_cublas_fc1_matmul_f32",
            "gac_cublas_fc2_matmul_f32",
        }:
            if (
                call.kind != "matmul"
                or call.implementation != "cupy-cublas"
                or len(call.inputs) != 2
                or call.grid is not None
                or call.block is not None
                or call.aliases_input is not None
                or call.transpose_a is not False
                or call.transpose_b is not False
                or call.alpha != 1.0
                or call.beta != 0.0
            ):
                raise BackendError(
                    f"MLIR runtime call {call.symbol!r} has an invalid MatMul ABI"
                )
            self._require_absent_parameters(
                call, ("rows", "channels", "elements", "epsilon")
            )
            left = graph.value_specs[call.inputs[0]]
            right = graph.value_specs[call.inputs[1]]
            output = graph.value_specs[call.output]
            if left.rank < 2 or right.rank != 2:
                raise BackendError(
                    f"MLIR runtime call {call.symbol!r} has unsupported MatMul ranks"
                )
            expected_m = math.prod(int(dim) for dim in left.shape[:-1])
            expected_k = int(left.shape[-1])
            expected_n = int(right.shape[-1])
            if (
                right.shape[0] != expected_k
                or output.shape != (*left.shape[:-1], expected_n)
                or (call.m, call.n, call.k)
                != (expected_m, expected_n, expected_k)
            ):
                raise BackendError(
                    f"MLIR runtime call {call.symbol!r} has inconsistent m/n/k or shapes"
                )
            return

        if call.symbol == "gac_bias_relu_inplace_f32":
            if (
                call.kind != "fused-kernel"
                or call.implementation != "nvrtc-cuda"
                or len(call.inputs) != 2
                or call.aliases_input != 0
            ):
                raise BackendError(
                    f"MLIR runtime call {call.symbol!r} has an invalid fusion ABI"
                )
            self._require_absent_parameters(
                call,
                (
                    "m",
                    "n",
                    "k",
                    "transpose_a",
                    "transpose_b",
                    "alpha",
                    "beta",
                    "epsilon",
                ),
            )
            value = graph.value_specs[call.inputs[0]]
            bias = graph.value_specs[call.inputs[1]]
            output = graph.value_specs[call.output]
            rows = math.prod(int(dim) for dim in value.shape[:-1])
            channels = int(value.shape[-1])
            elements = rows * channels
            if elements > CUDA_KERNEL_INDEX_MAX:
                raise BackendError(
                    f"MLIR runtime call {call.symbol!r} exceeds the signed "
                    "32-bit kernel index domain"
                )
            if call.grid is not None and call.grid[0] > CUDA_GRID_X_MAX:
                raise BackendError(
                    f"MLIR runtime call {call.symbol!r} exceeds the CUDA grid.x limit"
                )
            if (
                output != value
                or bias.shape != (channels,)
                or (call.rows, call.channels, call.elements)
                != (rows, channels, elements)
                or call.grid != ((elements + 255) // 256, 1, 1)
                or call.block != (256, 1, 1)
            ):
                raise BackendError(
                    f"MLIR runtime call {call.symbol!r} has inconsistent fusion shapes"
                )
            return

        if call.symbol == "gac_bias_residual_layer_norm_f32":
            if (
                call.kind != "fused-kernel"
                or call.implementation != "nvrtc-cuda"
                or len(call.inputs) != 5
                or call.aliases_input is not None
            ):
                raise BackendError(
                    f"MLIR runtime call {call.symbol!r} has an invalid LayerNorm ABI"
                )
            self._require_absent_parameters(
                call,
                (
                    "m",
                    "n",
                    "k",
                    "transpose_a",
                    "transpose_b",
                    "alpha",
                    "beta",
                ),
            )
            projected, bias, residual, gamma, beta = (
                graph.value_specs[name] for name in call.inputs
            )
            output = graph.value_specs[call.output]
            rows = math.prod(int(dim) for dim in output.shape[:-1])
            channels = int(output.shape[-1])
            elements = rows * channels
            if elements > CUDA_KERNEL_INDEX_MAX:
                raise BackendError(
                    f"MLIR runtime call {call.symbol!r} exceeds the signed "
                    "32-bit kernel index domain"
                )
            if call.grid is not None and call.grid[0] > CUDA_GRID_X_MAX:
                raise BackendError(
                    f"MLIR runtime call {call.symbol!r} exceeds the CUDA grid.x limit"
                )
            epsilon = call.epsilon
            if (
                projected != output
                or residual != output
                or bias.shape != (channels,)
                or gamma.shape != (channels,)
                or beta.shape != (channels,)
                or (call.rows, call.channels, call.elements)
                != (rows, channels, elements)
                or call.grid != (rows, 1, 1)
                or call.block != (256, 1, 1)
                or epsilon is None
                or not np.isfinite(np.float32(epsilon))
                or np.float32(epsilon) <= 0.0
            ):
                raise BackendError(
                    f"MLIR runtime call {call.symbol!r} has inconsistent LayerNorm shapes"
                )
            return

        raise BackendError(f"unknown MLIR runtime symbol {call.symbol!r}")

    def _validate_call_plan(self) -> None:
        validate_mlir_cuda_program(self.program)
        for call in self.program.calls:
            self._validate_call_abi(call)
            missing = [name for name in (*call.inputs, call.output) if name not in self._values]
            if missing:
                raise BackendError(
                    f"MLIR runtime call {call.symbol!r} references missing values {missing}"
                )
            if call.implementation == "nvrtc-cuda" and (
                call.grid is None or call.block is None or call.symbol not in self._kernels
            ):
                raise BackendError(
                    f"MLIR CUDA call {call.symbol!r} has no complete launch configuration"
                )
        if hasattr(self, "_graph"):
            self._validate_actual_values()

    def _call_info(self, call: MlirCudaCall) -> dict[str, Any]:
        info = call.to_dict()
        if call.symbol in self._kernels:
            info["attributes"] = {
                key: int(value)
                for key, value in self._kernels[call.symbol].attributes.items()
            }
        return info

    def _execute_call(self, call: MlirCudaCall) -> None:
        values = self._values
        if call.symbol == "gac_cublas_fc1_matmul_f32":
            self._cp.matmul(
                values[call.inputs[0]],
                values[call.inputs[1]],
                out=values[call.output],
            )
            return
        if call.symbol == "gac_cublas_fc2_matmul_f32":
            self._cp.matmul(
                values[call.inputs[0]],
                values[call.inputs[1]],
                out=values[call.output],
            )
            return
        if call.symbol == "gac_bias_relu_inplace_f32":
            assert call.grid is not None and call.block is not None
            self._kernels[call.symbol](
                call.grid,
                call.block,
                (values[call.inputs[0]], values[call.inputs[1]]),
                stream=self._stream,
            )
            return
        if call.symbol == "gac_bias_residual_layer_norm_f32":
            assert call.grid is not None and call.block is not None
            arguments = tuple(values[name] for name in call.inputs) + (
                values[call.output],
            )
            self._kernels[call.symbol](
                call.grid,
                call.block,
                arguments,
                stream=self._stream,
            )
            return
        raise BackendError(f"unknown MLIR runtime symbol {call.symbol!r}")

    def _execute_calls(self) -> None:
        """Submit the explicit call plan on the session stream.

        This private method intentionally performs no transfers or synchronization;
        benchmark tooling can wrap it in one outer CUDA Event pair without adding
        per-dispatch measurement overhead.
        """

        for call in self.program.calls:
            self._execute_call(call)

    def _upload_bound(self, bound: Mapping[str, np.ndarray]) -> None:
        for name, value in bound.items():
            self._inputs[name].set(value, stream=self._stream)

    def run(self, inputs: Mapping[str, np.ndarray]) -> RunResult:
        """Run one invocation including input upload and output download."""

        if not self._lock.acquire(blocking=False):
            raise BackendError(
                "concurrent calls on a single MLIR CUDA Session are not supported"
            )
        try:
            if self._closed or self._failed:
                raise BackendError(
                    "MLIR CUDA Session is closed or failed; create a new Session"
                )
            started = time.perf_counter()
            bound = bind_inputs(self._graph, inputs)
            with self._device, self._stream:
                start = self._cp.cuda.Event()
                end = self._cp.cuda.Event()
                start.record(self._stream)
                self._upload_bound(bound)
                self._execute_calls()
                end.record(self._stream)
                outputs = {
                    name: self._cp.asnumpy(
                        self._values[name], stream=self._stream, blocking=True
                    )
                    for name in self._graph.outputs
                }
                self._stream.synchronize()
                gpu_ms = float(self._cp.cuda.get_elapsed_time(start, end))
            return RunResult(
                outputs=outputs,
                wall_ms=(time.perf_counter() - started) * 1000.0,
                gpu_stream_ms=gpu_ms,
                kernel_ms=None,
                profiler=Profiler(enabled=False),
            )
        except (BackendError, ValidationError):
            raise
        except Exception as exc:
            self._failed = True
            try:
                with self._device:
                    self._stream.synchronize()
            except Exception:
                pass
            raise BackendError(f"MLIR CUDA invocation failed: {exc}") from exc
        finally:
            self._lock.release()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                if self._stream is not None:
                    with self._device:
                        self._stream.synchronize()
            finally:
                self._closed = True
                self._clear()

    def __enter__(self) -> "MlirCudaSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = ["MlirCudaSession"]
