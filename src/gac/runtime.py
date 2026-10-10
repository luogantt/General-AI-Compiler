"""Single-device CUDA execution and a separately labelled NumPy reference path."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import importlib
import importlib.metadata
import os
from pathlib import Path
from threading import Lock
import time
from typing import Any, Mapping

import numpy as np

from .backends.cuda import CudaProgram, KernelLaunch, NVRTC_OPTIONS, TARGET
from .errors import BackendError, DeviceUnavailableError, ValidationError
from .ir import Graph, Node
from .memory import plan_memory
from .ops import evaluate_node
from .profiler import Profiler


_cuda_library_handles: list[Any] = []


def _supports_cudnn_convolution(node: Node, graph: Graph) -> bool:
    """Return whether a Conv2D node can use CuPy's direct cuDNN path.

    cuDNN's classic 2-D convolution descriptor represents one symmetric pad per
    spatial axis.  Keep asymmetric convolutions on the generated CUDA kernel so
    choosing the vendor path never changes graph semantics.
    """
    if node.op != "Conv2D" or len(node.inputs) not in {2, 3}:
        return False
    input_spec = graph.value_specs[node.inputs[0]]
    weight_spec = graph.value_specs[node.inputs[1]]
    output_spec = graph.value_specs[node.outputs[0]]
    if (
        input_spec.dtype != "float32"
        or weight_spec.dtype != "float32"
        or output_spec.dtype != "float32"
        or len(input_spec.shape) != 4
        or len(weight_spec.shape) != 4
        or len(output_spec.shape) != 4
    ):
        return False
    pads = tuple(int(value) for value in node.attrs.get("pads", (0, 0, 0, 0)))
    return len(pads) == 4 and pads[0] == pads[2] and pads[1] == pads[3]


def _supports_cublas_matmul(node: Node, graph: Graph) -> bool:
    """Return whether CuPy can route a MatMul directly to cuBLAS."""
    if node.op != "MatMul" or len(node.inputs) != 2:
        return False
    left = graph.value_specs[node.inputs[0]]
    right = graph.value_specs[node.inputs[1]]
    output = graph.value_specs[node.outputs[0]]
    return (
        left.dtype == right.dtype == output.dtype == "float32"
        and len(left.shape) >= 2
        and len(right.shape) >= 2
        and len(output.shape) >= 2
    )


def _load_wheel_nvrtc() -> None:
    """CuPy 13 needs help locating pip-installed NVRTC outside the system path."""
    if _cuda_library_handles:
        return
    try:
        distribution = importlib.metadata.distribution("nvidia-cuda-nvrtc-cu12")
    except importlib.metadata.PackageNotFoundError:
        return
    root = Path(distribution.locate_file("nvidia/cuda_nvrtc"))
    if os.name == "nt":
        directory = root / "bin"
        if directory.is_dir():
            _cuda_library_handles.append(os.add_dll_directory(str(directory)))
    else:
        libraries = sorted((root / "lib").glob("libnvrtc.so.*"))
        if libraries:
            _cuda_library_handles.append(ctypes.CDLL(str(libraries[0]), mode=ctypes.RTLD_GLOBAL))


def _cupy():
    try:
        _load_wheel_nvrtc()
        return importlib.import_module("cupy")
    except (ImportError, OSError) as exc:
        raise DeviceUnavailableError(
            "CuPy/CUDA is unavailable. Install this project's [cuda12] extra in a "
            "separate environment on the NVIDIA machine. No CPU fallback was performed."
        ) from exc


def cuda_device_info(device_id: int = 0) -> dict[str, Any]:
    cp = _cupy()
    try:
        count = cp.cuda.runtime.getDeviceCount()
        if not 0 <= device_id < count:
            raise DeviceUnavailableError(f"CUDA device {device_id} does not exist (device count: {count})")
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        name = properties["name"]
        return {
            "device_id": device_id,
            "name": name.decode() if isinstance(name, bytes) else str(name),
            "compute_capability": [int(properties["major"]), int(properties["minor"])],
            "total_memory_bytes": int(properties["totalGlobalMem"]),
            "multiprocessor_count": int(properties["multiProcessorCount"]),
            "driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cupy_version": cp.__version__,
            "supported_target": TARGET if (properties["major"], properties["minor"]) == (8, 6) else None,
        }
    except DeviceUnavailableError:
        raise
    except Exception as exc:
        raise DeviceUnavailableError(f"Cannot query CUDA device {device_id}: {exc}") from exc


def bind_inputs(graph: Graph, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(inputs) != set(graph.inputs):
        raise ValidationError(f"input names must be {sorted(graph.inputs)}, got {sorted(inputs)}")
    bound = {}
    for name, spec in graph.inputs.items():
        value = np.asarray(inputs[name])
        if value.shape != spec.shape or value.dtype.name != spec.dtype:
            raise ValidationError(f"input {name!r} requires {spec.shape}/{spec.dtype}, got {value.shape}/{value.dtype}")
        if not value.flags.c_contiguous:
            raise ValidationError(f"input {name!r} must be C-contiguous")
        if not np.all(np.isfinite(value)):
            raise ValidationError(f"input {name!r} must contain finite values")
        bound[name] = value
    return bound


def reference_run(graph: Graph, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    values = dict(graph.constants)
    values.update(bind_inputs(graph, inputs))
    for node in graph.nodes:
        values.update(evaluate_node(node, values))
    return {name: values[name].copy() for name in graph.outputs}


@dataclass
class RunResult:
    outputs: dict[str, np.ndarray]
    wall_ms: float
    gpu_stream_ms: float
    kernel_ms: float | None
    profiler: Profiler


class CudaSession:
    """Owns a CUDA module, stream and resident constants; invocations are serial.

    All graph nodes run on the requested physical CUDA device. Missing devices,
    unsupported architectures, compilation failures and runtime failures raise.
    There is deliberately no implicit CPU execution path.
    """

    def __init__(
        self,
        program: CudaProgram,
        *,
        device_id: int = 0,
        conv_backend: str = "auto",
        cudnn_tensor_core: str = "never",
        matmul_backend: str = "auto",
    ) -> None:
        """Create one SM86 execution session.

        ``conv_backend`` accepts ``"auto"``, ``"cudnn"`` or ``"generated"``.
        Auto uses cuDNN for compatible FP32 Conv2D nodes when the CuPy binding is
        present and retains the generated CUDA implementation as a per-node
        fallback.  ``cudnn_tensor_core="never"`` preserves strict FP32 compute;
        callers can explicitly select ``"auto"`` or ``"always"`` for a faster,
        lower-precision cuDNN policy.  ``matmul_backend`` accepts ``"auto"``,
        ``"cupy"`` or ``"generated"``; the CuPy path writes into the static
        output buffer and uses the NVIDIA cuBLAS FP32 implementation.
        """
        if conv_backend not in {"auto", "cudnn", "generated"}:
            raise ValidationError(
                "conv_backend must be 'auto', 'cudnn', or 'generated'"
            )
        if cudnn_tensor_core not in {"never", "auto", "always"}:
            raise ValidationError(
                "cudnn_tensor_core must be 'never', 'auto', or 'always'"
            )
        if matmul_backend not in {"auto", "cupy", "generated"}:
            raise ValidationError(
                "matmul_backend must be 'auto', 'cupy', or 'generated'"
            )
        started = time.perf_counter()
        self.program = program
        self.info = cuda_device_info(device_id)
        if self.info["supported_target"] != TARGET:
            raise DeviceUnavailableError(f"{TARGET} requires compute capability 8.6, got {self.info['compute_capability']}")
        self._cp = _cupy()
        self._device = self._cp.cuda.Device(device_id)
        self._lock = Lock()
        self._closed = False
        self._failed = False
        self._constants: dict[str, Any] = {}
        self._inputs: dict[str, Any] = {}
        self._outputs: dict[str, Any] = {}
        self._arena_views: dict[str, Any] = {}
        self._arena_memory: Any = None
        self._kernels: dict[str, Any] = {}
        self._nodes_by_id = {node.id: node for node in program.graph.nodes}
        self._node_ops = {node.id: node.op for node in program.graph.nodes}
        self._cudnn: Any = None
        self._cudnn_launch_ids: set[str] = set()
        self._cudnn_tensor_core = cudnn_tensor_core
        self._cudnn_workspace_budget = 0
        self._cublas_launch_ids: set[str] = {
            launch.node_id
            for launch in program.launches
            if matmul_backend != "generated"
            and "cupy-cublas" in launch.implementation_candidates
            and _supports_cublas_matmul(
                self._nodes_by_id[launch.node_id], program.graph
            )
        }
        compatible_convolutions = {
            launch.node_id
            for launch in program.launches
            if "cudnn" in launch.implementation_candidates
            and _supports_cudnn_convolution(
                self._nodes_by_id[launch.node_id], program.graph
            )
        }
        if conv_backend != "generated" and compatible_convolutions:
            try:
                self._cudnn = importlib.import_module("cupyx.cudnn")
                if not callable(getattr(self._cudnn, "convolution_forward", None)):
                    raise ImportError("cupyx.cudnn.convolution_forward is unavailable")
                self._cudnn_launch_ids = compatible_convolutions
                workspace_query = getattr(
                    self._cudnn, "get_max_workspace_size", None
                )
                try:
                    workspace_budget = int(workspace_query())
                    if workspace_budget < 0:
                        raise ValueError("negative cuDNN workspace budget")
                    self._cudnn_workspace_budget = workspace_budget
                except (TypeError, ValueError, RuntimeError):
                    self._cudnn_workspace_budget = 8 * 1024 * 1024
            except (ImportError, OSError) as exc:
                if conv_backend == "cudnn":
                    raise DeviceUnavailableError(
                        "cuDNN convolution was requested but the CuPy cuDNN binding "
                        "is unavailable"
                    ) from exc
                self._cudnn = None
        self.info["convolution_backend"] = {
            "requested": conv_backend,
            "selected": "cudnn" if self._cudnn_launch_ids else "generated-cuda",
            "cudnn_launch_count": len(self._cudnn_launch_ids),
            "generated_launch_count": sum(
                self._node_ops[launch.node_id] == "Conv2D"
                and launch.node_id not in self._cudnn_launch_ids
                for launch in program.launches
            ),
            "tensor_core": (
                cudnn_tensor_core if self._cudnn_launch_ids else "not-applicable"
            ),
            "workspace_budget_bytes": self._cudnn_workspace_budget,
        }
        self.info["matmul_backend"] = {
            "requested": matmul_backend,
            "selected": (
                "cupy-cublas" if self._cublas_launch_ids else "generated-cuda"
            ),
            "cublas_launch_count": len(self._cublas_launch_ids),
            "generated_launch_count": sum(
                self._node_ops[launch.node_id] == "MatMul"
                and launch.node_id not in self._cublas_launch_ids
                for launch in program.launches
            ),
            "precision": "float32",
        }
        self.memory_plan = plan_memory(program.graph)
        try:
            with self._device:
                self._stream = self._cp.cuda.Stream(non_blocking=True)
                with self._stream:
                    planned_tensor_bytes = (
                        self.memory_plan.arena_size
                        + self.memory_plan.resident_bytes
                        + self.memory_plan.external_input_bytes
                        + self.memory_plan.external_output_bytes
                    )
                    required = (
                        planned_tensor_bytes + self._cudnn_workspace_budget
                    )
                    free, _ = self._cp.cuda.runtime.memGetInfo()
                    if required > free:
                        raise BackendError(f"planned tensor storage requires {required} bytes; only {free} CUDA bytes free")
                    self._module = self._cp.RawModule(code=program.source, options=NVRTC_OPTIONS, backend="nvrtc")
                    self._module.compile()
                    self._kernels = {launch.entry_point: self._module.get_function(launch.entry_point) for launch in program.launches}
                    self.kernel_info = [
                        {"entry_point": launch.entry_point, "node_id": launch.node_id,
                         "op": self._node_ops[launch.node_id],
                         "implementation": self._implementation_for(launch),
                         "implementation_candidates": list(
                             launch.implementation_candidates
                         ),
                         "source_nodes": list(launch.source_nodes),
                         "grid": list(launch.grid), "block": list(launch.block),
                         "attributes": {key: int(value) for key, value in self._kernels[launch.entry_point].attributes.items()}}
                        for launch in program.launches
                    ]
                    for name, value in program.graph.constants.items():
                        dtype = program.graph.value_specs[name].dtype
                        self._constants[name] = self._cp.array(value, dtype=dtype, order="C", copy=True)
                    for name, spec in program.graph.inputs.items():
                        self._inputs[name] = self._cp.empty(spec.shape, dtype=spec.dtype)
                    for name in program.graph.outputs:
                        if name not in self._inputs and name not in self._constants:
                            spec = program.graph.value_specs[name]
                            self._outputs[name] = self._cp.empty(spec.shape, dtype=spec.dtype)
                    if self.memory_plan.arena_size:
                        self._arena_memory = self._cp.cuda.alloc(self.memory_plan.arena_size)
                        for allocation in self.memory_plan.allocations:
                            if allocation.storage != "arena":
                                continue
                            assert allocation.offset is not None
                            pointer = self._arena_memory + allocation.offset
                            self._arena_views[allocation.name] = self._cp.ndarray(
                                allocation.spec.shape,
                                dtype=allocation.spec.dtype,
                                memptr=pointer,
                            )
                    self.info["memory_plan"] = {
                        "arena_size": self.memory_plan.arena_size,
                        "peak_live_bytes": self.memory_plan.peak_live_bytes,
                        "resident_bytes": self.memory_plan.resident_bytes,
                        "external_input_bytes": self.memory_plan.external_input_bytes,
                        "external_output_bytes": self.memory_plan.external_output_bytes,
                        "planned_device_bytes": planned_tensor_bytes,
                        "vendor_workspace_budget_bytes": self._cudnn_workspace_budget,
                        "required_device_bytes_upper_bound": required,
                    }
                    self._stream.synchronize()
        except Exception as exc:
            self._constants.clear()
            self._inputs.clear()
            self._outputs.clear()
            self._arena_views.clear()
            self._arena_memory = None
            self._kernels.clear()
            self._closed = True
            raise BackendError(f"CUDA session initialization failed: {exc}") from exc
        self.setup_ms = (time.perf_counter() - started) * 1000

    def _implementation_for(self, launch: KernelLaunch) -> str:
        if launch.node_id in getattr(self, "_cudnn_launch_ids", ()):
            return "cudnn"
        if launch.node_id in getattr(self, "_cublas_launch_ids", ()):
            return "cupy-cublas"
        return "generated-cuda"

    def _execute_launch(
        self,
        launch: KernelLaunch,
        values: Mapping[str, Any],
    ) -> None:
        args = tuple(values[name] for name in launch.inputs) + (
            values[launch.output],
        )
        implementation = self._implementation_for(launch)
        if implementation == "cupy-cublas":
            self._cp.matmul(args[0], args[1], out=args[-1])
            return
        if implementation != "cudnn":
            self._kernels[launch.entry_point](
                launch.grid, launch.block, args, stream=self._stream
            )
            return

        node = self._nodes_by_id[launch.node_id]
        pads = tuple(int(value) for value in node.attrs.get("pads", (0, 0, 0, 0)))
        strides = tuple(
            int(value) for value in node.attrs.get("strides", (1, 1))
        )
        dilations = tuple(
            int(value) for value in node.attrs.get("dilations", (1, 1))
        )
        groups = int(node.attrs.get("groups", 1))
        bias = args[2] if len(launch.inputs) == 3 else None
        self._cudnn.convolution_forward(
            args[0],
            args[1],
            bias,
            args[-1],
            (pads[0], pads[1]),
            strides,
            dilations,
            groups,
            auto_tune=False,
            tensor_core=self._cudnn_tensor_core,
        )

    def run(
        self,
        inputs: Mapping[str, np.ndarray],
        *,
        trace: bool = False,
        profile: bool = False,
    ) -> RunResult:
        """Execute one invocation.

        ``profile`` records one CUDA Event pair around every launch.  ``trace``
        keeps its existing behavior and also enables those device timings so a
        Chrome trace can carry them.  Both options are off by default because
        per-launch Events perturb the workload they measure.
        """
        if not self._lock.acquire(blocking=False):
            raise BackendError("concurrent calls on a single CUDA Session are not supported")
        try:
            if self._closed or self._failed:
                raise BackendError("CUDA Session is closed or failed; create a new Session")
            started = time.perf_counter()
            bound = bind_inputs(self.program.graph, inputs)
            cp = self._cp
            instrument_launches = trace or profile
            profiler = Profiler(enabled=instrument_launches)
            event_pairs = []
            with self._device, self._stream:
                values = dict(self._constants)
                values.update(self._inputs)
                values.update(self._outputs)
                values.update(self._arena_views)
                start, end = cp.cuda.Event(), cp.cuda.Event()
                start.record(self._stream)
                with profiler.record("copy_h2d", "inputs", TARGET, metadata={"bytes": sum(v.nbytes for v in bound.values()), "clock": "host submission"}):
                    for name, value in bound.items():
                        self._inputs[name].set(value, stream=self._stream)
                for launch in self.program.launches:
                    spec = self.program.graph.value_specs[launch.output]
                    allocation = self.memory_plan.allocation(launch.output)
                    pair = (
                        (cp.cuda.Event(), cp.cuda.Event())
                        if instrument_launches
                        else None
                    )
                    metadata = {"node_id": launch.node_id, "entry_point": launch.entry_point,
                                "op": self._node_ops[launch.node_id],
                                "implementation": self._implementation_for(launch),
                                "grid": launch.grid, "block": launch.block, "clock": "host submission",
                                "output_bytes": spec.nbytes, "storage": allocation.storage,
                                "arena_offset": allocation.offset}
                    if pair:
                        pair[0].record(self._stream)
                    with profiler.record("dispatch", launch.entry_point, TARGET, source_nodes=launch.source_nodes, metadata=metadata):
                        self._execute_launch(launch, values)
                    if pair:
                        pair[1].record(self._stream)
                        event_pairs.append((pair, profiler.events[-1].metadata))
                end.record(self._stream)
                with profiler.record("copy_d2h", "outputs", TARGET, metadata={"clock": "host including completion wait"}):
                    outputs = {name: cp.asnumpy(values[name], stream=self._stream, blocking=True) for name in self.program.graph.outputs}
                self._stream.synchronize()
                gpu_ms = float(cp.cuda.get_elapsed_time(start, end))
                kernel_ms = None
                if instrument_launches:
                    kernel_ms = 0.0
                    for pair, metadata in event_pairs:
                        elapsed = float(cp.cuda.get_elapsed_time(*pair))
                        metadata["gpu_elapsed_ms"] = elapsed
                        kernel_ms += elapsed
            return RunResult(outputs, (time.perf_counter() - started) * 1000, gpu_ms, kernel_ms, profiler)
        except ValidationError:
            raise
        except BackendError:
            raise
        except Exception as exc:
            self._failed = True
            with self._device:
                try:
                    self._stream.synchronize()
                except Exception:
                    pass
            raise BackendError(f"CUDA invocation failed; Session is invalidated: {exc}") from exc
        finally:
            self._lock.release()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                try:
                    with self._device:
                        self._stream.synchronize()
                finally:
                    self._constants.clear()
                    self._inputs.clear()
                    self._outputs.clear()
                    self._arena_views.clear()
                    self._arena_memory = None
                    self._kernels.clear()
                    self._cudnn_launch_ids.clear()
                    self._cublas_launch_ids.clear()
                    self._cudnn = None
                    self._module = None
                    self._closed = True

    def __enter__(self) -> "CudaSession":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
