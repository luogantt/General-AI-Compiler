"""CPU-only checks of the experiment installer's dispatch and rollback contract."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / "tools" / "galaxeavla" / "g05_npu_graphs.py"
SPEC = importlib.util.spec_from_file_location("g05_npu_graphs", PATH)
GRAPHS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = GRAPHS
SPEC.loader.exec_module(GRAPHS)


class GraphInstallerTests(unittest.TestCase):
    def make_runtime(self, fail_at=None):
        events = []

        def compile_method(original, **options):
            events.append((original, options))
            if len(events) == fail_at:
                raise RuntimeError("compiler construction failed")

            def compiled(*args, **kwargs):
                return "compiled", original(*args, **kwargs)

            return compiled

        torch = SimpleNamespace(
            compile=compile_method,
            _dynamo=SimpleNamespace(config=SimpleNamespace(suppress_errors=True)),
        )
        return torch, events

    def make_source(self):
        return SimpleNamespace(**{
            name: (lambda *args, **kwargs: (args, kwargs))
            for name in GRAPHS.GDN_HELPERS.values()
        })

    def test_gdn_dispatch_preserves_arguments_and_other_devices(self):
        torch, calls = self.make_runtime()
        source = self.make_source()
        original = source.torch_chunk_gated_delta_rule
        state = object()
        with patch.object(GRAPHS.importlib, "import_module", return_value=source):
            handle = GRAPHS.install_gdn_backends(torch, helpers=("chunk",))
        npu_query = SimpleNamespace(device=SimpleNamespace(type="npu"))
        cpu_query = SimpleNamespace(device=SimpleNamespace(type="cpu"))
        cuda_query = SimpleNamespace(device=SimpleNamespace(type="cuda"))
        kwargs = {"initial_state": state, "chunk_size": 32, "output_final_state": True}
        self.assertEqual(source.torch_chunk_gated_delta_rule(npu_query, **kwargs),
                         ("compiled", ((npu_query,), kwargs)))
        self.assertEqual(source.torch_chunk_gated_delta_rule(query=npu_query, **kwargs),
                         ("compiled", ((), {"query": npu_query, **kwargs})))
        for query in (cpu_query, cuda_query):
            self.assertEqual(source.torch_chunk_gated_delta_rule(query, **kwargs),
                             original(query, **kwargs))
        self.assertEqual(calls[0][1]["options"], {"clone_input": True, "clone_output": True})
        handle.restore()
        handle.restore()
        self.assertIs(source.torch_chunk_gated_delta_rule, original)

    def test_partial_installation_is_rolled_back_when_compile_fails(self):
        torch, _ = self.make_runtime(fail_at=2)
        source = self.make_source()
        originals = vars(source).copy()
        with patch.object(GRAPHS.importlib, "import_module", return_value=source):
            with self.assertRaisesRegex(RuntimeError, "compiler construction failed"):
                GRAPHS.install_gdn_backends(torch)
        for name, original in originals.items():
            self.assertIs(getattr(source, name), original)

    def test_restore_does_not_clobber_a_later_external_change(self):
        torch, _ = self.make_runtime()
        source = self.make_source()
        with patch.object(GRAPHS.importlib, "import_module", return_value=source):
            handle = GRAPHS.install_gdn_backends(torch, helpers=("recurrent",))
        replacement = lambda *args: None
        source.torch_recurrent_gated_delta_rule = replacement
        handle.restore()
        self.assertIs(source.torch_recurrent_gated_delta_rule, replacement)

    def test_unsafe_cache_options_and_silent_noops_are_rejected(self):
        torch, calls = self.make_runtime()
        for option in ("clone_input", "clone_output"):
            with self.assertRaisesRegex(ValueError, "preserve cache values"):
                GRAPHS.install_gdn_backends(torch, options={option: False})
        for options in ({"static_kernel_compile": 1}, {"unknown": True}):
            with self.assertRaises((TypeError, ValueError)):
                GRAPHS.install_gdn_backends(torch, options=options)
        with self.assertRaisesRegex(ValueError, "require npugraph_ex"):
            GRAPHS.install_gdn_backends(torch, "eager", options={"frozen_parameter": True})
        self.assertFalse(calls)

    def test_module_compile_keeps_other_methods_and_reports_selected_options(self):
        torch, calls = self.make_runtime()
        module = SimpleNamespace(forward=lambda x: x, embed=object(), decode=object())
        before = vars(module).copy()
        options = {"frozen_parameter": True, "static_kernel_compile": True,
                   "super_kernel_optimize": True}
        original, metadata = GRAPHS.install_module_backend(
            torch, module, "npugraph_ex", "AE.forward", options=options,
        )
        self.assertIs(original, before["forward"])
        self.assertIs(module.embed, before["embed"])
        self.assertIs(module.decode, before["decode"])
        self.assertEqual(calls[0][1]["options"], options)
        self.assertTrue(metadata["compilation_is_lazy"])
        self.assertFalse(torch._dynamo.config.suppress_errors)

    def test_ge_frozen_parameter_uses_compiler_config_not_compile_options(self):
        for frozen in (True, False):
            with self.subTest(frozen_parameter=frozen):
                torch, calls = self.make_runtime()
                module = SimpleNamespace(forward=lambda x: x)
                config = SimpleNamespace(
                    mode=SimpleNamespace(value="max-autotune"),
                    experimental_config=SimpleNamespace(frozen_parameter=False),
                )
                backend = object()
                backend_configs = []

                def make_backend(*, compiler_config):
                    backend_configs.append(compiler_config)
                    return backend

                imports = {
                    "torchair": SimpleNamespace(get_npu_backend=make_backend),
                    "torchair.configs.compiler_config": SimpleNamespace(
                        CompilerConfig=lambda: config,
                    ),
                }
                with patch.object(GRAPHS.importlib, "import_module", side_effect=imports.__getitem__):
                    _, metadata = GRAPHS.install_module_backend(
                        torch, module, "torchair", "AE.forward",
                        options={"frozen_parameter": frozen},
                    )
                self.assertEqual(backend_configs, [config])
                self.assertIs(config.experimental_config.frozen_parameter, frozen)
                self.assertIs(calls[0][1]["backend"], backend)
                self.assertNotIn("options", calls[0][1])
                self.assertEqual(metadata["options"], {"frozen_parameter": frozen})
                self.assertEqual(metadata["compiler_config_options"], {
                    "experimental_config.frozen_parameter": frozen,
                })

    def test_ge_rejects_other_options_and_eager_rejects_all_options(self):
        torch, calls = self.make_runtime()
        module = SimpleNamespace(forward=lambda x: x)
        for option in ("static_kernel_compile", "super_kernel_optimize", "clone_input", "clone_output"):
            with self.subTest(option=option):
                with self.assertRaisesRegex(ValueError, "Unsupported TorchAir GE"):
                    GRAPHS.install_module_backend(
                        torch, module, "torchair", "AE.forward", options={option: True},
                    )
        with self.assertRaisesRegex(TypeError, "must be a bool"):
            GRAPHS.install_module_backend(
                torch, module, "torchair", "AE.forward", options={"frozen_parameter": 1},
            )
        for options in ({"frozen_parameter": True}, {"frozen_parameter": False}):
            with self.assertRaisesRegex(ValueError, "not eager"):
                GRAPHS.install_module_backend(torch, module, "eager", "AE.forward", options=options)
        self.assertFalse(calls)

    def test_softplus_decomposition_is_local_to_one_ge_backend(self):
        torch, calls = self.make_runtime()
        softplus_operator = object()
        eye_operator = object()
        torch.ops = SimpleNamespace(aten=SimpleNamespace(
            softplus=SimpleNamespace(default=softplus_operator),
            eye=SimpleNamespace(default=eye_operator),
        ))
        configs = []
        backend_kwargs = []

        def make_config():
            config = SimpleNamespace(
                mode=SimpleNamespace(value="max-autotune"),
                experimental_config=SimpleNamespace(frozen_parameter=False),
                ge_config=SimpleNamespace(enable_single_stream=False),
            )
            configs.append(config)
            return config

        def make_backend(**kwargs):
            backend_kwargs.append(kwargs)
            return object()

        imports = {
            "torchair": SimpleNamespace(get_npu_backend=make_backend),
            "torchair.configs.compiler_config": SimpleNamespace(CompilerConfig=make_config),
        }
        with patch.object(GRAPHS.importlib, "import_module", side_effect=imports.__getitem__):
            _, enabled = GRAPHS.install_module_backend(
                torch, SimpleNamespace(forward=lambda x: x), "torchair", "VLM.forward",
                options={"frozen_parameter": True, "softplus_decomposition": True,
                         "single_stream": True, "eye_decomposition": True},
            )
            _, disabled = GRAPHS.install_module_backend(
                torch, SimpleNamespace(forward=lambda x: x), "torchair", "AE.forward",
            )
        self.assertEqual(set(backend_kwargs[0]["custom_decompositions"]), {softplus_operator, eye_operator})
        self.assertTrue(callable(backend_kwargs[0]["custom_decompositions"][softplus_operator]))
        self.assertTrue(callable(backend_kwargs[0]["custom_decompositions"][eye_operator]))
        self.assertNotIn("custom_decompositions", backend_kwargs[1])
        self.assertFalse(hasattr(configs[0].experimental_config, "softplus_decomposition"))
        self.assertTrue(configs[0].experimental_config.frozen_parameter)
        self.assertTrue(configs[0].ge_config.enable_single_stream)
        self.assertFalse(configs[1].ge_config.enable_single_stream)
        self.assertTrue(enabled["compiler_config_options"]["ge_config.enable_single_stream"])
        self.assertTrue(enabled["softplus_decomposition"]["enabled"])
        self.assertTrue(enabled["softplus_decomposition"]["rounding_order_changed"])
        self.assertFalse(enabled["softplus_decomposition"]["global_registration_modified"])
        self.assertFalse(disabled["softplus_decomposition"]["enabled"])
        self.assertFalse(disabled["softplus_decomposition"]["rounding_order_changed"])
        self.assertTrue(enabled["eye_decomposition"]["enabled"])
        self.assertFalse(enabled["eye_decomposition"]["global_registration_modified"])
        self.assertFalse(disabled["eye_decomposition"]["enabled"])
        for _, compile_kwargs in calls:
            self.assertNotIn("options", compile_kwargs)

    def test_joint_option_parser_does_not_relax_backend_validation(self):
        self.assertEqual(GRAPHS.validate_graph_options({"softplus_decomposition": True}),
                         {"softplus_decomposition": True})
        torch, calls = self.make_runtime()
        with self.assertRaisesRegex(ValueError, "Unsupported NPU graph"):
            GRAPHS.install_module_backend(
                torch, SimpleNamespace(forward=lambda x: x), "npugraph_ex", "AE.forward",
                options={"softplus_decomposition": True},
            )
        with self.assertRaisesRegex(ValueError, "Unsupported NPU graph"):
            GRAPHS.validate_npugraph_options({"softplus_decomposition": False})
        with self.assertRaisesRegex(TypeError, "must be a bool"):
            GRAPHS.validate_graph_options({"softplus_decomposition": 1})
        self.assertFalse(calls)

    def test_eye_option_is_ge_only(self):
        self.assertEqual(GRAPHS.validate_graph_options({"eye_decomposition": True}),
                         {"eye_decomposition": True})
        torch, calls = self.make_runtime()
        for name in ("npugraph_ex", "eager"):
            with self.subTest(backend=name), self.assertRaises(ValueError):
                GRAPHS.install_module_backend(
                    torch, SimpleNamespace(forward=lambda x: x), name, "VLM.forward",
                    options={"eye_decomposition": True},
                )
        with self.assertRaisesRegex(TypeError, "must be a bool"):
            GRAPHS.validate_torchair_options({"eye_decomposition": 1})
        self.assertFalse(calls)

    def test_torchair_aclgraph_and_ge_share_entry_point_but_not_config(self):
        class Config:
            # The real NpuBaseConfig setter updates OptionValue.value.
            def __init__(self):
                self._mode = SimpleNamespace(value="max-autotune")
                self.experimental_config = SimpleNamespace(frozen_parameter=False)

            @property
            def mode(self):
                return self._mode

            @mode.setter
            def mode(self, value):
                self._mode.value = value

        for order in (("torchair", "torchair_aclgraph"),
                      ("torchair_aclgraph", "torchair")):
            for frozen in (True, False):
                with self.subTest(order=order, frozen_parameter=frozen):
                    torch, calls = self.make_runtime()
                    configs = []
                    backends = []
                    metadata = []

                    def make_backend(**kwargs):
                        self.assertEqual(set(kwargs), {"compiler_config"})
                        configs.append(kwargs["compiler_config"])
                        backend = object()
                        backends.append(backend)
                        return backend

                    imports = {
                        "torchair": SimpleNamespace(get_npu_backend=make_backend),
                        "torchair.configs.compiler_config": SimpleNamespace(CompilerConfig=Config),
                    }
                    with patch.object(GRAPHS.importlib, "import_module",
                                      side_effect=imports.__getitem__) as importer:
                        for name in order:
                            _, info = GRAPHS.install_module_backend(
                                torch, SimpleNamespace(forward=lambda x: x), name,
                                "AE.forward" if name == "torchair" else "VLM.forward",
                                options={"frozen_parameter": frozen},
                            )
                            metadata.append(info)
                    self.assertIsNot(configs[0], configs[1])
                    self.assertEqual({call.args[0] for call in importer.call_args_list}, set(imports))
                    for name, config, backend, info, (_, kwargs) in zip(
                            order, configs, backends, metadata, calls):
                        expected_mode = "npugraph_ex" if name == "torchair_aclgraph" else "max-autotune"
                        self.assertEqual(config.mode.value, expected_mode)
                        self.assertEqual(info["mode"], expected_mode)
                        self.assertIs(config.experimental_config.frozen_parameter, frozen)
                        self.assertIs(kwargs["backend"], backend)
                        self.assertNotIn("options", kwargs)
                        self.assertFalse(info["fallback_allowed"])
                        if name == "torchair_aclgraph":
                            self.assertTrue(info["experimental"])
                            self.assertFalse(info["independent_npugraph_ex_backend_requested"])
                            self.assertEqual(info["compiler_config_options"], {
                                "mode": "npugraph_ex",
                                "experimental_config.frozen_parameter": frozen,
                            })

    def test_torchair_aclgraph_rejects_unverified_or_ge_only_options(self):
        torch, calls = self.make_runtime()
        for option in ("static_kernel_compile", "super_kernel_optimize", "clone_input",
                       "clone_output", "single_stream", "softplus_decomposition", "eye_decomposition"):
            for value in (True, False):
                with self.subTest(option=option, value=value):
                    with self.assertRaisesRegex(ValueError, "Unsupported TorchAir ACL graph"):
                        GRAPHS.install_module_backend(
                            torch, SimpleNamespace(forward=lambda x: x), "torchair_aclgraph",
                            "VLM.forward", options={option: value},
                        )
        with self.assertRaisesRegex(TypeError, "must be a bool"):
            GRAPHS.validate_torchair_aclgraph_options({"frozen_parameter": 1})
        self.assertFalse(calls)


if __name__ == "__main__":
    unittest.main()
