"""Opt-in, process-local NPU graph experiments for the pinned G0.5 source.

Install after loading the model and before warmup. Nothing is written to the
upstream checkout or site-packages. Compilation errors remain fatal; installation
itself does not execute a model or prove that its first compilation will succeed.

The GDN helpers retain the original equations, casts, FP32 recurrent state and
chunk size. They compile tensor-only functions instead of the whole VLM forward,
avoiding its mutable SparseKVCache and variable autoregressive cache lengths.
Prefill shapes, initial_state=None versus Tensor, and final-state flags can still
produce separate specializations. Do not stack this experiment with whole-VLM
compilation when measuring its individual effect.

``torchair_aclgraph`` is a separate opt-in experiment: it selects the installed
TorchAir package's own ``npugraph_ex`` mode through ``get_npu_backend``. It does
not import the independent npugraph_ex package. Mixed GE/ACL execution still
requires device-side correctness and timing validation; installation proves
neither, and this path has not been NPU-validated by this helper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import functools
import importlib
from typing import Any, Callable, Mapping, Sequence


GDN_MODULE = "g05.models.g05.qwen35.gated_deltanet"
GDN_HELPERS = {
    "chunk": "torch_chunk_gated_delta_rule",
    "recurrent": "torch_recurrent_gated_delta_rule",
    "causal-conv": "torch_causal_conv1d_update",
}
_BOOLEAN_OPTIONS = frozenset({
    "frozen_parameter", "static_kernel_compile", "super_kernel_optimize",
    "clone_input", "clone_output",
})
_GE_BOOLEAN_OPTIONS = frozenset({
    "frozen_parameter", "softplus_decomposition", "single_stream", "eye_decomposition",
})
_TORCHAIR_ACL_BOOLEAN_OPTIONS = frozenset({"frozen_parameter"})
_INSTALL_MARKER = "_g05_npu_graph_installation"


def _validate_boolean_options(
    options: Mapping[str, Any] | None, supported: frozenset[str], backend: str,
) -> dict[str, bool]:
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("NPU graph options must be a mapping")
    unknown = set(options) - supported
    if unknown:
        raise ValueError(f"Unsupported {backend} options: {sorted(unknown)}")
    result = dict(options)
    for name, value in result.items():
        if type(value) is not bool:
            raise TypeError(f"NPU graph option {name!r} must be a bool")
    return result


def validate_npugraph_options(options: Mapping[str, Any] | None) -> dict[str, bool]:
    """Validate the small option subset supported by the installed CANN 9 image."""
    return _validate_boolean_options(options, _BOOLEAN_OPTIONS, "NPU graph")


def validate_torchair_options(options: Mapping[str, Any] | None) -> dict[str, bool]:
    """Only expose GE settings verified in the pinned TorchNPU image."""
    return _validate_boolean_options(options, _GE_BOOLEAN_OPTIONS, "TorchAir GE")


def validate_torchair_aclgraph_options(options: Mapping[str, Any] | None) -> dict[str, bool]:
    """Restrict the same-package ACL experiment to the verified config field."""
    return _validate_boolean_options(options, _TORCHAIR_ACL_BOOLEAN_OPTIONS, "TorchAir ACL graph")


def validate_graph_options(options: Mapping[str, Any] | None) -> dict[str, bool]:
    """CLI parsing only; installation still validates the selected backend."""
    return _validate_boolean_options(options, _BOOLEAN_OPTIONS | _GE_BOOLEAN_OPTIONS, "NPU graph")


def make_softplus_decomposition(torch: Any) -> Callable:
    """Build the optional GE-only FP32 softplus decomposition without registration.

    This is limited to GDN's default beta/threshold and preserves the linear
    threshold branch. The stable identity changes floating-point rounding, so
    enabling it requires numerical validation; it is not bitwise-equivalent.
    """
    def softplus_fp32(x, beta=1, threshold=20):
        if x.dtype != torch.float32:
            raise TypeError("G0.5 softplus decomposition requires a float32 input")
        if (not isinstance(beta, (int, float)) or isinstance(beta, bool) or beta != 1
                or not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
                or threshold != 20):
            raise ValueError("G0.5 softplus decomposition requires beta=1 and threshold=20")
        # All inputs and elementwise intermediates remain FP32. exp(-abs(x))
        # cannot overflow for finite x; where retains PyTorch's threshold rule.
        return torch.where(
            x > 20, x, torch.relu(x) + torch.log1p(torch.exp(-torch.abs(x))),
        )

    return softplus_fp32


def make_eye_decomposition(torch: Any) -> Callable:
    """Build aten.eye.default's square identity using backend-local tensor ops.

    Layout and pinning requests are never silently discarded. The CPU default
    dtype follows torch.get_default_dtype(); explicit device/dtype are retained.
    This function does not implement the separate rectangular aten.eye.m overload.
    """
    def eye_square(n, *, dtype=None, layout=None, device=None, pin_memory=None):
        if layout is not None and layout != torch.strided:
            raise RuntimeError("G0.5 eye decomposition supports only strided layout")
        if pin_memory is not None and type(pin_memory) is not bool:
            raise TypeError("pin_memory must be a bool or None")
        if n < 0:
            raise RuntimeError(f"n must be greater or equal to 0, got {n}")
        indices = torch.arange(n, device=device)
        result = (indices.unsqueeze(0) == indices.unsqueeze(1)).to(
            dtype=dtype if dtype is not None else torch.get_default_dtype(),
        )
        if pin_memory:
            # Pin the output, not just the temporary index tensor. This raises
            # for non-CPU output or an unavailable pinned-memory allocator,
            # matching eye's allocation constraints instead of ignoring them.
            result = result.pin_memory()
        return result

    return eye_square


def _compile_callable(
    torch: Any,
    original: Callable,
    name: str,
    scope: str,
    options: Mapping[str, Any] | None,
) -> tuple[Callable, dict[str, Any]]:
    if name not in {"eager", "npugraph_ex", "torchair", "torchair_aclgraph"}:
        raise ValueError(f"Unknown NPU backend: {name}")
    if name == "eager" and options:
        raise ValueError("Graph options require a compiled backend, not eager")
    selected_options = (
        validate_torchair_options(options) if name == "torchair"
        else validate_torchair_aclgraph_options(options) if name == "torchair_aclgraph"
        else validate_npugraph_options(options)
    )
    metadata = {
        "name": name, "scope": scope,
        "fullgraph": True, "dynamic": False, "fallback_allowed": False,
        "options": selected_options,
        "compilation_is_lazy": name != "eager",
    }
    if name == "eager":
        metadata.update(fullgraph=None, dynamic=None)
        return original, metadata
    if getattr(original, _INSTALL_MARKER, False):
        raise RuntimeError(f"Already installed a G0.5 NPU graph at {scope}")
    # Never turn a backend failure into an apparently successful eager result.
    torch._dynamo.config.suppress_errors = False
    if name in {"torchair", "torchair_aclgraph"}:
        # Both paths stay inside one TorchAir package. Do not mix either with
        # the independent npugraph_ex backend in this image: its distributed
        # operator registration can collide with TorchAir's registrations.
        torchair = importlib.import_module("torchair")
        config_module = importlib.import_module("torchair.configs.compiler_config")
        config = config_module.CompilerConfig()
        config_options = {}
        if name == "torchair_aclgraph":
            # Installed npu_fx_compiler._gen_compiled_gm selects its own
            # AclConcreteGraph for this mode, not the separate backend package.
            config.mode = "npugraph_ex"
            config_options["mode"] = "npugraph_ex"
        if "frozen_parameter" in selected_options:
            value = selected_options["frozen_parameter"]
            config.experimental_config.frozen_parameter = value
            config_options["experimental_config.frozen_parameter"] = value
        if "single_stream" in selected_options:
            value = selected_options["single_stream"]
            config.ge_config.enable_single_stream = value
            config_options["ge_config.enable_single_stream"] = value
        softplus_enabled = selected_options.get("softplus_decomposition", False)
        eye_enabled = selected_options.get("eye_decomposition", False)
        backend_kwargs = {"compiler_config": config}
        custom_decompositions = {}
        if softplus_enabled:
            # Backend-local dictionary only: no torch._decomp/global converter
            # registration and no source/site-packages modification.
            custom_decompositions[torch.ops.aten.softplus.default] = make_softplus_decomposition(torch)
        if eye_enabled:
            custom_decompositions[torch.ops.aten.eye.default] = make_eye_decomposition(torch)
        if custom_decompositions:
            backend_kwargs["custom_decompositions"] = custom_decompositions
        backend = torchair.get_npu_backend(**backend_kwargs)
        metadata["backend_module"] = getattr(torchair, "__file__", None)
        metadata["mode"] = config.mode.value
        metadata["compiler_config_options"] = config_options
        if name == "torchair_aclgraph":
            metadata["experimental"] = True
            metadata["runtime_validation"] = "not performed by installer; NPU validation required"
            metadata["backend_entry_point"] = "torchair.get_npu_backend"
            metadata["concrete_graph"] = "torchair._acl_concrete_graph.AclConcreteGraph"
            metadata["independent_npugraph_ex_backend_requested"] = False
        metadata["softplus_decomposition"] = {
            "enabled": softplus_enabled, "default_enabled": False,
            "operator": "aten.softplus.default", "input_and_math_dtype": "float32",
            "beta": 1, "threshold": 20,
            "formula": "where(x > 20, x, relu(x) + log1p(exp(-abs(x))))",
            "rounding_order_changed": softplus_enabled,
            "global_registration_modified": False,
        }
        metadata["eye_decomposition"] = {
            "enabled": eye_enabled, "default_enabled": False,
            "operator": "aten.eye.default", "scope": "square identity only",
            "formula": "(arange(n)[:, None] == arange(n)[None, :]).to(requested_dtype)",
            "values": "same exact zero/one identity entries",
            "default_dtype": "torch.get_default_dtype()",
            "layout": "strided; reject other layouts",
            "pin_memory": "honor requested output pinning; propagate allocation errors",
            "global_registration_modified": False,
        }
    else:
        backend = "npugraph_ex"
    compile_kwargs = {"backend": backend, "fullgraph": True, "dynamic": False}
    if selected_options and name == "npugraph_ex":
        compile_kwargs["options"] = selected_options
    compiled = torch.compile(original, **compile_kwargs)
    setattr(compiled, _INSTALL_MARKER, True)
    return compiled, metadata


def install_module_backend(
    torch: Any,
    module: Any,
    name: str,
    scope: str,
    *,
    options: Mapping[str, Any] | None = None,
) -> tuple[Callable, dict[str, Any]]:
    """Compile only forward, retaining embed/decode and other module methods.

    Returns the original forward and metadata. Restore with
    ``module.forward = original`` in the caller's finally block.
    """
    original = module.forward
    compiled, metadata = _compile_callable(torch, original, name, scope, options)
    if name != "eager":
        module.forward = compiled
    return original, metadata


def install_ae_backend(
    torch: Any,
    policy: Any,
    name: str,
    *,
    options: Mapping[str, Any] | None = None,
) -> tuple[Callable, dict[str, Any]]:
    return install_module_backend(
        torch, policy.model.action_expert, name,
        "policy.model.action_expert.forward only", options=options,
    )


@dataclass
class GraphInstallation:
    """A reversible set of module-global function replacements.

    restore() is idempotent and preserves any later replacement installed by
    someone else. It does not clear Dynamo caches or modify model/cache tensors.
    """

    metadata: dict[str, Any]
    _replacements: list[tuple[Any, str, Callable, Callable]] = field(
        default_factory=list, repr=False,
    )
    _restored: bool = field(default=False, init=False, repr=False)

    def restore(self) -> None:
        if self._restored:
            return
        for module, name, original, installed in reversed(self._replacements):
            if getattr(module, name) is installed:
                setattr(module, name, original)
        self._restored = True

    def __enter__(self) -> "GraphInstallation":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.restore()


def _npu_dispatch(original: Callable, compiled: Callable, input_name: str) -> Callable:
    @functools.wraps(original)
    def dispatch(*args, **kwargs):
        tensor = args[0] if args else kwargs.get(input_name)
        device = getattr(tensor, "device", None)
        if getattr(device, "type", None) == "npu":
            return compiled(*args, **kwargs)
        return original(*args, **kwargs)

    setattr(dispatch, _INSTALL_MARKER, True)
    return dispatch


def install_gdn_backends(
    torch: Any,
    name: str = "npugraph_ex",
    *,
    helpers: Sequence[str] = ("chunk", "recurrent"),
    options: Mapping[str, Any] | None = None,
) -> GraphInstallation:
    """Compile selected GDN module globals used by the upstream NPU branch.

    One helper graph is reused across layers. Its returned FP32 recurrent states
    must remain independent after subsequent replays, so clone_output=True is
    required. clone_input=True is also required as layer/request inputs have
    different allocations. This preserves cache values at a measurable copy cost.

    The optional causal-conv helper mutates conv_state; the graph backend must
    preserve that mutation. Verify consecutive decode steps before using it.
    Only NPU inputs dispatch to compiled helpers; CPU/CUDA behavior is unchanged.
    """
    if name not in {"eager", "npugraph_ex"}:
        raise ValueError("GDN helper backend must be eager or npugraph_ex")
    if isinstance(helpers, str):
        raise TypeError("helpers must be a sequence of names, not a comma-separated string")
    selected = tuple(helpers)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select at least one GDN helper without duplicates")
    unknown = set(selected) - set(GDN_HELPERS)
    if unknown:
        raise ValueError(f"Unknown GDN helpers: {sorted(unknown)}")
    selected_options = validate_npugraph_options(options)
    if name == "eager" and selected_options:
        raise ValueError("GDN graph options require npugraph_ex")
    if name != "eager":
        for option in ("clone_input", "clone_output"):
            if selected_options.get(option) is False:
                raise ValueError(f"GDN helper graphs require {option}=True to preserve cache values")
            selected_options[option] = True
    installation = GraphInstallation(metadata={
        "name": name,
        "scope": "NPU GatedDeltaNet tensor helpers; all layers in this process",
        "module": GDN_MODULE,
        "selected_helpers": list(selected),
        "options": selected_options,
        "helpers": {},
        "equations_changed": False,
        "recurrent_state_precision": "original FP32",
        "chunk_size": "unchanged; upstream caller selects 32",
        "source_files_modified": False,
        "npu_only": True,
        "fallback_allowed": False,
    })
    if name == "eager":
        return installation
    module = importlib.import_module(GDN_MODULE)
    try:
        for helper in selected:
            attribute = GDN_HELPERS[helper]
            original = getattr(module, attribute)
            compiled, metadata = _compile_callable(
                torch, original, name, f"{GDN_MODULE}.{attribute}", selected_options,
            )
            input_name = "hidden_states" if helper == "causal-conv" else "query"
            dispatch = _npu_dispatch(original, compiled, input_name)
            installation._replacements.append((module, attribute, original, dispatch))
            setattr(module, attribute, dispatch)
            installation.metadata["helpers"][helper] = metadata
    except Exception:
        installation.restore()
        raise
    return installation
