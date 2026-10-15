"""Executable dependency boundaries for the canonical package architecture.

AST inspection includes imports inside functions and TYPE_CHECKING blocks.
Compatibility aliases exist for callers; implementation modules must use their
canonical paths so those aliases cannot hide a reversed layer dependency.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "gac"


def _module(path: Path) -> str:
    parts = path.relative_to(SOURCE).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(("gac", *parts))


def _imports(tree: ast.AST, module: str, *, package: bool, known: set[str]):
    """Yield (line, module) for static and literal dynamic local imports."""
    owner = module if package else module.rsplit(".", 1)[0]
    dynamic_names = {"import_module", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "importlib":
            dynamic_names.update(alias.asname or alias.name for alias in node.names if alias.name == "import_module")
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level:
                name = importlib.util.resolve_name("." * node.level + name, owner)
            names = [name]
            names.extend(name + "." + alias.name for alias in node.names if name + "." + alias.name in known)
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            dynamic = (isinstance(function, ast.Name) and function.id in dynamic_names) or (
                isinstance(function, ast.Attribute) and function.attr == "import_module"
            )
            if dynamic and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                name = node.args[0].value
                if name.startswith("."):
                    package_arg = next((k.value for k in node.keywords if k.arg == "package"), None)
                    if package_arg is None and len(node.args) > 1:
                        package_arg = node.args[1]
                    if isinstance(package_arg, ast.Constant) and isinstance(package_arg.value, str):
                        name = importlib.util.resolve_name(name, package_arg.value)
                    elif isinstance(package_arg, ast.Name) and package_arg.id == "__package__":
                        name = importlib.util.resolve_name(name, owner)
                names = [name]
        for name in names:
            if name == "gac" or name.startswith("gac."):
                yield node.lineno, name


def _within(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


class ArchitectureLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.aliases = importlib.import_module("gac._compat").LEGACY_MODULE_ALIASES
        cls.modules = {_module(path): path for path in SOURCE.rglob("*.py")}
        known = set(cls.modules) | set(cls.aliases)
        cls.dependencies = {
            name: tuple(_imports(ast.parse(path.read_text(encoding="utf-8")), name,
                                 package=path.name == "__init__.py", known=known))
            for name, path in cls.modules.items()
        }

    def assert_boundary(self, prefix: str, allowed: tuple[str, ...]):
        violations = []
        for module, imports in self.dependencies.items():
            if not _within(module, prefix):
                continue
            for line, dependency in imports:
                if dependency != "gac" and not any(_within(dependency, group) for group in allowed):
                    violations.append(f"{self.modules[module].relative_to(ROOT)}:{line}: {dependency}")
        self.assertEqual(violations, [], "Layer dependencies must point toward contracts:\n" + "\n".join(violations))

    def test_core_has_no_upper_layer_dependencies(self):
        self.assert_boundary("gac.core", ("gac.core",))

    def test_generic_compiler_depends_only_on_core_and_itself(self):
        self.assert_boundary("gac.compiler", ("gac.core", "gac.compiler"))

    def test_generic_frontends_do_not_select_models_or_devices(self):
        self.assert_boundary("gac.frontends", ("gac.core", "gac.compiler", "gac.frontends"))

    def test_reference_validation_does_not_depend_on_device_execution(self):
        self.assert_boundary("gac.validation", ("gac.core", "gac.validation"))

    def test_model_builders_do_not_import_execution_or_targets(self):
        violations = []
        forbidden = ("gac.targets", "gac.execution", "gac.workflows", "gac.tools", "gac.tuning")
        for module, imports in self.dependencies.items():
            if not _within(module, "gac.models"):
                continue
            for line, dependency in imports:
                if any(_within(dependency, group) for group in forbidden):
                    violations.append(f"{module}:{line}: {dependency}")
        self.assertEqual(violations, [], "Model graph construction must be device independent:\n" + "\n".join(violations))

    def test_common_model_builders_do_not_depend_on_concrete_models(self):
        self.assert_boundary("gac.models.common", ("gac.core", "gac.compiler", "gac.models.common"))

    def test_model_families_share_common_code_without_importing_each_other(self):
        violations = []
        for module, imports in self.dependencies.items():
            if not module.startswith("gac.models."):
                continue
            family = module.split(".")[2]
            if family in {"api", "common"}:
                continue
            for line, dependency in imports:
                if dependency.startswith("gac.models.") and dependency.split(".")[2] not in {family, "api", "common"}:
                    violations.append(f"{module}:{line}: {dependency}")
        self.assertEqual(violations, [], "Share model components through models/common:\n" + "\n".join(violations))

    def test_targets_do_not_import_concrete_models(self):
        violations = [
            f"{module}:{line}: {dependency}"
            for module, imports in self.dependencies.items() if _within(module, "gac.targets")
            for line, dependency in imports if _within(dependency, "gac.models")
        ]
        self.assertEqual(violations, [], "Targets consume Graph contracts rather than model builders:\n" + "\n".join(violations))

    def test_internal_imports_never_use_legacy_aliases(self):
        violations = []
        for module, imports in self.dependencies.items():
            if module == "gac._compat" or _within(module, "gac.backends"):
                continue
            for line, dependency in imports:
                if any(_within(dependency, old) for old in self.aliases):
                    violations.append(f"{module}:{line}: {dependency}")
        self.assertEqual(violations, [], "Compatibility aliases are for external callers only:\n" + "\n".join(violations))

    def test_alias_destinations_have_physical_canonical_modules(self):
        missing = {old: new for old, new in self.aliases.items() if new not in self.modules}
        self.assertEqual(missing, {})
        self.assertFalse(set(self.aliases) & set(self.modules), "A virtual legacy alias must not shadow a physical canonical module")

    def test_root_import_is_lazy_in_a_fresh_process(self):
        code = """import json, sys
import gac
blocked = ('torch', 'cupy', 'onnx', 'gac.targets', 'gac.models', 'gac.execution')
loaded = sorted(name for name in sys.modules if any(name == p or name.startswith(p + '.') for p in blocked))
print(json.dumps(loaded))
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True, timeout=30, check=True)
        self.assertEqual(json.loads(result.stdout), [], "import gac must not initialize optional models, compilers or device runtimes")

    def test_dependency_scan_includes_local_and_dynamic_imports(self):
        source = """def deferred():
    from ...targets.vortex import compiler
    import importlib
    importlib.import_module('gac.models.bevformer.ffn')
"""
        imports = {name for _, name in _imports(ast.parse(source), "gac.models.bevformer.probe", package=False,
                                               known={"gac.targets.vortex.compiler"})}
        self.assertIn("gac.targets.vortex.compiler", imports)
        self.assertIn("gac.models.bevformer.ffn", imports)


if __name__ == "__main__":
    unittest.main()
