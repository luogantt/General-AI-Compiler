"""CUDA runtime for MLIR-generated SM86 PTX artifacts.

The compiler owns the PTX and launch ABI.  This module only validates the
complete artifact, binds statically-shaped device buffers, loads that PTX, and
submits the declared kernels.  It never recompiles CUDA source and it has no
cuBLAS, NVRTC, NumPy, or legacy-CUDA execution fallback.
"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
import tempfile
from threading import Lock
import time
from typing import Any, Mapping

import numpy as np

from gac.targets.cuda.ptx import (
    COMPUTE_CAPABILITY,
    RUNTIME_ABI_VERSION,
    TARGET,
    MlirPtxLaunch,
    MlirPtxProgram,
    validate_mlir_ptx_program,
)
from gac.core.errors import BackendError, DeviceUnavailableError, ValidationError
from gac.execution.profiler import Profiler
from gac.execution.cuda import RunResult, _cupy, cuda_device_info
from gac.validation.reference import bind_inputs


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class MlirPtxSession:
    """Execute one validated MLIR-generated PTX program on an SM86 GPU.

    A session owns its CUDA device, non-blocking stream, loaded module, resident
    constants, inputs, outputs, and intermediate buffers.  Calls are serialized
    because these buffers are reused across invocations.
    """

    def __init__(self, program: MlirPtxProgram, *, device_id: int = 0) -> None:
        if not isinstance(program, MlirPtxProgram):
            raise ValidationError("program must be an MlirPtxProgram")

        # This check is deliberately the first operation.  In particular, a
        # malformed or modified artifact must fail before CUDA discovery imports
        # CuPy, creates a context, writes executable PTX, or allocates memory.
        validate_mlir_ptx_program(program)

        started = time.perf_counter()
        self.program = program
        self._graph = program.graph.clone()
        self._lock = Lock()
        self._closed = False
        self._failed = False
        self._cp: Any | None = None
        self._device: Any | None = None
        self._stream: Any | None = None
        self._module: Any | None = None
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._ptx_path: Path | None = None
        self._constants: dict[str, Any] = {}
        self._inputs: dict[str, Any] = {}
        self._outputs: dict[str, Any] = {}
        self._workspaces: dict[str, Any] = {}
        self._values: dict[str, Any] = {}
        self._kernels: dict[str, Any] = {}

        # cuda_device_info is the first CUDA-touching operation and is reached
        # only after all artifact/hash/PTX/ABI checks above have succeeded.
        self.info = cuda_device_info(device_id)
        if tuple(self.info["compute_capability"]) != COMPUTE_CAPABILITY:
            raise DeviceUnavailableError(
                f"{TARGET} requires compute capability 8.6, got "
                f"{self.info['compute_capability']}"
            )
        self.info["compiler_backend"] = TARGET
        self.info["runtime_abi_version"] = RUNTIME_ABI_VERSION
        self.info["runtime_dispatch_source"] = "mlir-generated-ptx"
        self.info["matmul_backend"] = {
            "selected": "mlir-generated-ptx",
            "cublas_launch_count": 0,
            "nvrtc_launch_count": 0,
            "precision": "float32",
        }

        self._cp = _cupy()
        self._device = self._cp.cuda.Device(device_id)
        try:
            with self._device:
                self._stream = self._cp.cuda.Stream(non_blocking=True)
                with self._stream:
                    self._load_module()
                    self._check_memory_budget()
                    self._allocate_values()
                    self._validate_actual_values()
                    self._stream.synchronize()
                    self.kernel_info = [
                        self._launch_info(launch) for launch in program.launches
                    ]
        except BaseException as exc:
            self._closed = True
            try:
                if self._stream is not None and self._device is not None:
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
            raise BackendError(f"MLIR PTX session initialization failed: {exc}") from exc

        self.setup_ms = (time.perf_counter() - started) * 1000.0

    def _load_module(self) -> None:
        assert self._cp is not None
        ptx_bytes = self.program.ptx.encode("utf-8")
        expected_digest = _sha256_text(self.program.ptx)
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="gac-mlir-ptx-"
        )
        self._ptx_path = Path(self._temporary_directory.name) / "program-sm86.ptx"
        self._ptx_path.write_bytes(ptx_bytes)
        actual_digest = hashlib.sha256(self._ptx_path.read_bytes()).hexdigest()
        if not hmac.compare_digest(actual_digest, expected_digest):
            raise BackendError("temporary MLIR PTX file failed its SHA-256 check")

        # RawModule's documented path mode asks the CUDA Driver to load an
        # existing PTX/cubin.  Supplying neither code nor a backend is important:
        # this path must never invoke NVRTC.
        self._module = self._cp.RawModule(path=str(self._ptx_path))
        for launch in self.program.launches:
            try:
                self._kernels[launch.symbol] = self._module.get_function(
                    launch.symbol
                )
            except Exception as exc:
                raise BackendError(
                    f"MLIR PTX entry {launch.symbol!r} could not be loaded: {exc}"
                ) from exc
        if set(self._kernels) != {launch.symbol for launch in self.program.launches}:
            raise BackendError("MLIR PTX module did not resolve the exact launch symbols")

    def _check_memory_budget(self) -> None:
        assert self._cp is not None
        names = set(self._graph.constants) | set(self._graph.inputs)
        names.update(launch.output for launch in self.program.launches)
        try:
            required = sum(self._graph.value_specs[name].nbytes for name in names)
        except KeyError as exc:
            raise BackendError(
                f"MLIR PTX launch references an unspecified value {exc.args[0]!r}"
            ) from exc
        free, _ = self._cp.cuda.runtime.memGetInfo()
        if required > int(free):
            raise BackendError(
                f"MLIR PTX tensor storage requires {required} bytes; "
                f"only {int(free)} CUDA bytes free"
            )
        self.info["memory_plan"] = {
            "planned_device_bytes": int(required),
            "required_device_bytes_upper_bound": int(required),
        }

    def _allocate_values(self) -> None:
        assert self._cp is not None
        graph = self._graph
        for name, value in graph.constants.items():
            dtype = graph.value_specs[name].dtype
            self._constants[name] = self._cp.array(
                value, dtype=dtype, order="C", copy=True
            )
        for name, spec in graph.inputs.items():
            self._inputs[name] = self._cp.empty(
                spec.shape, dtype=spec.dtype, order="C"
            )
        for name in graph.outputs:
            spec = graph.value_specs[name]
            self._outputs[name] = self._cp.empty(
                spec.shape, dtype=spec.dtype, order="C"
            )

        self._values = dict(self._constants)
        self._values.update(self._inputs)
        for launch in self.program.launches:
            missing = [name for name in launch.inputs if name not in self._values]
            if missing:
                raise BackendError(
                    f"MLIR PTX launch {launch.symbol!r} uses values before definition: "
                    f"{missing}"
                )
            if launch.output in self._values:
                raise BackendError(
                    f"MLIR PTX launch {launch.symbol!r} redefines {launch.output!r}"
                )
            spec = graph.value_specs.get(launch.output)
            if spec is None:
                raise BackendError(
                    f"MLIR PTX launch {launch.symbol!r} has no output tensor spec"
                )
            value = self._outputs.get(launch.output)
            if value is None:
                value = self._cp.empty(spec.shape, dtype=spec.dtype, order="C")
                self._workspaces[launch.output] = value
            self._values[launch.output] = value

        missing_outputs = [
            name
            for name in graph.outputs
            if self._values.get(name) is not self._outputs[name]
        ]
        if missing_outputs:
            raise BackendError(
                f"MLIR PTX launch plan does not define graph outputs {missing_outputs}"
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
            for launch in self.program.launches
            for name in (*launch.inputs, launch.output)
        }
        for name in referenced:
            if name not in self._values or name not in graph.value_specs:
                raise BackendError(f"MLIR PTX value {name!r} is not bound")
            value = self._values[name]
            spec = graph.value_specs[name]
            try:
                shape = tuple(int(dimension) for dimension in value.shape)
                dtype = np.dtype(value.dtype).name
            except (AttributeError, TypeError, ValueError) as exc:
                raise BackendError(f"MLIR PTX value {name!r} has no tensor ABI") from exc
            if shape != spec.shape or dtype != spec.dtype:
                raise BackendError(
                    f"MLIR PTX value {name!r} requires {spec.shape}/{spec.dtype}, "
                    f"got {shape}/{dtype}"
                )
            if not self._is_c_contiguous(value):
                raise BackendError(f"MLIR PTX value {name!r} is not C-contiguous")
            pointer = self._device_pointer(value)
            if pointer is not None and pointer == 0:
                raise BackendError(f"MLIR PTX value {name!r} has a null device pointer")

        # The ABI declares no aliases.  Enforce both Python-object identity and,
        # when exposed by the array object, the underlying device pointer.
        for launch in self.program.launches:
            output = self._values[launch.output]
            output_pointer = self._device_pointer(output)
            for input_name in launch.inputs:
                input_value = self._values[input_name]
                if output is input_value:
                    raise BackendError(
                        f"MLIR PTX launch {launch.symbol!r} has an undeclared object alias"
                    )
                input_pointer = self._device_pointer(input_value)
                if (
                    output_pointer is not None
                    and input_pointer is not None
                    and output_pointer == input_pointer
                ):
                    raise BackendError(
                        f"MLIR PTX launch {launch.symbol!r} has an undeclared pointer alias"
                    )

    def _launch_info(self, launch: MlirPtxLaunch) -> dict[str, Any]:
        info = launch.to_dict()
        kernel = self._kernels[launch.symbol]
        try:
            attributes = {
                key: int(value) for key, value in kernel.attributes.items()
            }
        except (AttributeError, TypeError, ValueError):
            attributes = {}
        info["attributes"] = attributes
        maximum = attributes.get("max_threads_per_block")
        threads = int(launch.block[0]) * int(launch.block[1]) * int(launch.block[2])
        if maximum is not None and threads > maximum:
            raise BackendError(
                f"MLIR PTX launch {launch.symbol!r} requests {threads} threads, "
                f"but the loaded kernel allows {maximum}"
            )
        return info

    def _execute_launch(self, launch: MlirPtxLaunch) -> None:
        arguments = tuple(self._values[name] for name in launch.inputs) + (
            self._values[launch.output],
        )
        self._kernels[launch.symbol](
            launch.grid,
            launch.block,
            arguments,
            stream=self._stream,
        )

    def _execute_launches(self) -> None:
        """Submit all PTX entries without transfers or synchronization."""

        for launch in self.program.launches:
            self._execute_launch(launch)

    def _upload_bound(self, bound: Mapping[str, np.ndarray]) -> None:
        for name, value in bound.items():
            self._inputs[name].set(value, stream=self._stream)

    def run(self, inputs: Mapping[str, np.ndarray]) -> RunResult:
        """Run one invocation, including input upload and output download."""

        if not self._lock.acquire(blocking=False):
            raise BackendError(
                "concurrent calls on a single MLIR PTX Session are not supported"
            )
        try:
            if self._closed or self._failed:
                raise BackendError(
                    "MLIR PTX Session is closed or failed; create a new Session"
                )
            started = time.perf_counter()
            bound = bind_inputs(self._graph, inputs)
            assert self._cp is not None
            assert self._device is not None
            assert self._stream is not None
            with self._device, self._stream:
                start = self._cp.cuda.Event()
                end = self._cp.cuda.Event()
                start.record(self._stream)
                self._upload_bound(bound)
                self._execute_launches()
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
                if self._device is not None and self._stream is not None:
                    with self._device:
                        self._stream.synchronize()
            except Exception:
                pass
            raise BackendError(f"MLIR PTX invocation failed: {exc}") from exc
        finally:
            self._lock.release()

    def _clear(self) -> None:
        self._values.clear()
        self._workspaces.clear()
        self._outputs.clear()
        self._inputs.clear()
        self._constants.clear()
        self._kernels.clear()
        self._module = None
        self._stream = None
        self._device = None
        self._cp = None
        self._ptx_path = None
        temporary_directory = self._temporary_directory
        self._temporary_directory = None
        if temporary_directory is not None:
            temporary_directory.cleanup()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                if self._stream is not None and self._device is not None:
                    with self._device:
                        self._stream.synchronize()
            finally:
                self._closed = True
                self._clear()

    def __enter__(self) -> "MlirPtxSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = ["MlirPtxSession"]
