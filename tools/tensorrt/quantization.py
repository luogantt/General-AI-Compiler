"""Strict TensorRT 10 PTQ data/cache adapter; no model preprocessing or Torch.

The directory must contain calibration.json, for example::

    {"schema_version": 1, "source_kind": "real_preprocessed",
     "dataset": "nuScenes calibration subset, actual camera frames",
     "preprocessing": "Describe the exact exporter-compatible preprocessing",
     "files": ["frame_000.npy", "frame_001.npz"]}

Each file is one FP32 batch, normally [1, 6, 3, 928, 1600]; .npz has exactly
one key, the network input name. No normalization, casting or reshaping occurs.
The declaration documents provenance; it does not prove that pixels are real
or that the selected frames represent the deployment distribution.

create_calibrator() requires a driver with malloc(nbytes)->int,
copy_host_to_device(pointer, contiguous_numpy_array), and free(pointer).
The caller owns the current CUDA context and implements these operations using
CUDA Driver APIs. Copies must complete before returning. Keep the calibrator
alive through build_serialized_network(), call assert_complete() on success,
and close() in finally. No device allocation occurs on a valid cache hit.

This uses deprecated but available TRT10 IInt8EntropyCalibrator2. It is not a
TRT11/Q-DQ implementation. Calibration is not model accuracy validation.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


DEFAULT_INPUT_SHAPE = (1, 6, 3, 928, 1600)
CACHE_MAGIC = b"GAC_TRT_ENTROPY2_CACHE_V1\n"


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value, label):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    return value.lower()


def read_input(path, input_name, input_shape):
    """Load one preprocessed sample without changing its mathematical values."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False, mmap_mode="r")
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if archive.files != [input_name]:
                raise ValueError(f"{path}: expected only key {input_name!r}, got {archive.files}")
            array = archive[input_name]
    else:
        raise ValueError(f"Calibration input must be .npy or .npz: {path}")
    if array.dtype != np.dtype("float32") or tuple(array.shape) != tuple(input_shape):
        raise ValueError(f"{path}: expected FP32 {tuple(input_shape)}, got {array.dtype} {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{path}: calibration input contains NaN or infinity")
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class CalibrationDataset:
    directory: Path
    input_name: str
    input_shape: tuple
    provenance: dict
    samples: tuple

    @classmethod
    def load(cls, directory, *, input_name="images", input_shape=DEFAULT_INPUT_SHAPE,
             allow_synthetic_for_testing=False):
        root = Path(directory).resolve()
        manifest = root / "calibration.json"
        if not manifest.is_file():
            raise ValueError(f"Missing {manifest}; declare provenance and an explicit calibration file list")
        provenance = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(provenance, dict) or provenance.get("schema_version") != 1:
            raise ValueError("calibration.json must be an object with schema_version=1")
        kind = provenance.get("source_kind")
        if kind != "real_preprocessed":
            if not (kind == "synthetic_interface_test" and allow_synthetic_for_testing):
                raise ValueError("Calibration requires source_kind=real_preprocessed; random inputs are not real calibration")
        for field in ("dataset", "preprocessing"):
            if not isinstance(provenance.get(field), str) or not provenance[field].strip():
                raise ValueError(f"calibration.json must describe {field}")
        names = provenance.get("files")
        if not isinstance(names, list) or not names or not all(isinstance(name, str) and name for name in names):
            raise ValueError("calibration.json files must be a nonempty explicit list of relative .npy/.npz paths")
        shape = tuple(input_shape)
        if not shape or shape[0] != 1 or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in shape):
            raise ValueError("Calibration requires positive static dimensions with batch=1")
        if not allow_synthetic_for_testing and shape != DEFAULT_INPUT_SHAPE:
            raise ValueError(f"BEVFormer Base calibration requires {DEFAULT_INPUT_SHAPE}; other shapes are only for interface tests")
        samples, seen = [], set()
        for name in names:
            relative = Path(name)
            path = (root / relative).resolve()
            if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Calibration file must exist inside its directory: {name}")
            if path in seen:
                raise ValueError(f"Duplicate calibration sample: {name}")
            seen.add(path)
            read_input(path, input_name, shape)  # Fail before invoking any GPU callback.
            samples.append({"file": path.relative_to(root).as_posix(), "sha256": _file_sha256(path),
                            "bytes": path.stat().st_size})
        return cls(root, input_name, shape, provenance, tuple(samples))

    def batch(self, index):
        sample = self.samples[index]
        path = self.directory / sample["file"]
        if _file_sha256(path) != sample["sha256"]:
            raise ValueError(f"Calibration input changed after preflight: {path}")
        return read_input(path, self.input_name, self.input_shape)

    def binding(self):
        return {"input_name": self.input_name, "input_shape": list(self.input_shape), "input_dtype": "float32",
                "provenance": self.provenance, "ordered_samples": list(self.samples)}


class CalibrationCache:
    """Bind opaque TRT cache bytes to the exact model, policy, inputs and TRT version."""

    def __init__(self, path, binding):
        self.path = Path(path)
        self.binding = binding
        self.fingerprint = hashlib.sha256(_json_bytes(binding)).hexdigest()

    def read(self):
        if not self.path.exists():
            return None
        raw = self.path.read_bytes()
        if not raw.startswith(CACHE_MAGIC):
            raise ValueError(f"Unbound or invalid calibration cache: {self.path}")
        header_bytes, separator, payload = raw[len(CACHE_MAGIC):].partition(b"\n")
        if not separator or not payload:
            raise ValueError(f"Truncated calibration cache: {self.path}")
        header = json.loads(header_bytes)
        if (header.get("schema_version") != 1 or header.get("fingerprint") != self.fingerprint
                or header.get("binding") != self.binding):
            raise ValueError("Calibration cache does not match model/policy/data/version; choose a new cache path")
        if header.get("payload_sha256") != hashlib.sha256(payload).hexdigest():
            raise ValueError("Calibration cache payload checksum differs")
        return payload

    def write(self, payload):
        payload = bytes(payload)
        if not payload:
            raise ValueError("TensorRT returned an empty calibration cache")
        self.read()  # Refuse to replace a cache belonging to a different experiment.
        header = {"schema_version": 1, "fingerprint": self.fingerprint, "binding": self.binding,
                  "payload_sha256": hashlib.sha256(payload).hexdigest()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix=self.path.name + ".", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(CACHE_MAGIC + _json_bytes(header) + b"\n" + payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def create_calibrator(trt, *, directory, model_sha256, policy_fingerprint, input_name="images",
                      input_shape=DEFAULT_INPUT_SHAPE, cache_path, driver,
                      allow_synthetic_for_testing=False):
    """Return (TRT entropy calibrator, mutable audit metadata); caller must close it."""
    if int(trt.__version__.split(".")[0]) != 10 or not hasattr(trt, "IInt8EntropyCalibrator2"):
        raise ValueError("Legacy PTQ requires TensorRT 10 IInt8EntropyCalibrator2")
    for method in ("malloc", "copy_host_to_device", "free"):
        if not callable(getattr(driver, method, None)):
            raise TypeError(f"Calibration CUDA driver must provide {method}()")
    dataset = CalibrationDataset.load(directory, input_name=input_name, input_shape=input_shape,
                                      allow_synthetic_for_testing=allow_synthetic_for_testing)
    binding = {"model_sha256": _require_sha256(model_sha256, "model_sha256"),
               "policy_fingerprint": _require_sha256(policy_fingerprint, "policy_fingerprint"),
               "tensorrt": trt.__version__, "algorithm": "IInt8EntropyCalibrator2",
               "calibration": dataset.binding()}
    cache = CalibrationCache(cache_path, binding)
    cached_payload = cache.read()  # Reject cache mismatches before TensorRT can swallow callback errors.
    metadata = {"scope": "Activation-range calibration for the supplied precision policy; not accuracy validation",
                "dataset": dataset.binding(), "source_directory": str(dataset.directory),
                "fingerprint": cache.fingerprint, "cache_path": str(cache.path.resolve()),
                "model_sha256": binding["model_sha256"], "policy_fingerprint": binding["policy_fingerprint"],
                "tensorrt": trt.__version__, "algorithm": binding["algorithm"],
                "api_status": "legacy PTQ, deprecated in TensorRT 10; unsupported in TensorRT 11",
                "source_kind": dataset.provenance["source_kind"],
                "synthetic_interface_test": dataset.provenance["source_kind"] == "synthetic_interface_test",
                "sample_count": len(dataset.samples), "consumed_batches": 0,
                "cache_hit": False, "cache_written": False, "error": None}

    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.pointer = None
            self.closed = False
            self.host_batch = None

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            try:
                if self.closed:
                    raise RuntimeError("Calibration buffer has already been released")
                if list(names) != [input_name]:
                    raise ValueError(f"Expected only calibration input {input_name!r}; TensorRT requested {list(names)}")
                index = metadata["consumed_batches"]
                if index == len(dataset.samples):
                    return None
                self.host_batch = dataset.batch(index)
                if self.pointer is None:
                    pointer = int(driver.malloc(self.host_batch.nbytes))
                    if pointer == 0:
                        raise RuntimeError("CUDA calibration allocation returned a null pointer")
                    self.pointer = pointer
                driver.copy_host_to_device(self.pointer, self.host_batch)
                metadata["consumed_batches"] += 1
                print(f"Calibration input {metadata['consumed_batches']}/{len(dataset.samples)} submitted", flush=True)
                return [self.pointer]
            except Exception as error:
                metadata["error"] = f"{type(error).__name__}: {error}"
                raise

        def read_calibration_cache(self):
            metadata["cache_hit"] = cached_payload is not None
            return cached_payload

        def write_calibration_cache(self, payload):
            try:
                self.assert_complete()
                cache.write(payload)
                metadata["cache_written"] = True
            except Exception as error:
                metadata["error"] = f"{type(error).__name__}: {error}"
                raise

        def raise_if_error(self):
            if metadata["error"]:
                raise RuntimeError(metadata["error"])

        def assert_complete(self):
            self.raise_if_error()
            if not metadata["cache_hit"] and metadata["consumed_batches"] != len(dataset.samples):
                raise RuntimeError("TensorRT did not consume the declared calibration set or reuse its matching cache")

        def close(self):
            if self.pointer is not None:
                driver.free(self.pointer)
                self.pointer = None
            self.host_batch = None
            self.closed = True

    return EntropyCalibrator(), metadata
