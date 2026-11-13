from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import call, patch

from gac.targets.ascend.probe import probe_ascend_environment
from gac.tools.probe_ascend import main


def _command(name, arguments):
    del arguments
    if name == "npu-smi":
        return {
            "available": True,
            "ok": True,
            "executable": "/usr/local/bin/npu-smi",
            "returncode": 0,
            "stdout": "Name: Ascend 910B4",
            "stderr": "",
        }
    return {
        "available": True,
        "ok": True,
        "executable": "/usr/local/Ascend/atc",
        "returncode": 0,
        "stdout": "ATC version 9",
        "stderr": "",
    }


def _runtime(product="Ascend 910B4"):
    return {
        "importable": True,
        "torch_version": "2.7.1",
        "torch_npu_version": "2.7.1.post1",
        "device_count": 1,
        "device_names": [product],
        "triton_importable": True,
        "triton_version": "3.6.0",
        "triton_target": {"backend": "npu", "arch": "Ascend910B4", "warp_size": 0},
        "error": None,
    }


class AscendProbeTests(unittest.TestCase):
    @patch("gac.targets.ascend.probe._existing_toolkit_roots", return_value=("/usr/local/Ascend/latest",))
    @patch("gac.targets.ascend.probe._python_runtime_probe", return_value=_runtime())
    @patch("gac.targets.ascend.probe._command_probe", side_effect=_command)
    @patch("gac.targets.ascend.probe.platform.machine", return_value="aarch64")
    @patch("gac.targets.ascend.probe.platform.system", return_value="Linux")
    def test_reports_confirmed_910b_runtime_without_claiming_gac_execution(
        self, _system, _machine, _commands, _runtime_probe, _roots
    ):
        report = probe_ascend_environment(environ={}, python_executable="/opt/g05/bin/python")
        self.assertTrue(report["readiness"]["torch_npu_discoverable"])
        self.assertTrue(report["readiness"]["triton_ascend_discoverable"])
        self.assertFalse(report["readiness"]["torch_npu_execution_verified"])
        self.assertFalse(report["readiness"]["triton_ascend_compile_verified"])
        self.assertFalse(report["readiness"]["gac_npubin_compiler_ready"])
        self.assertFalse(report["readiness"]["gac_physical_runtime_ready"])
        self.assertIn("Ascend 910B4", report["hardware"]["products"])
        self.assertEqual(report["host"]["python_executable"], "/opt/g05/bin/python")
        self.assertIn(call("atc", ("--help",)), _commands.call_args_list)

    @patch("gac.targets.ascend.probe._existing_toolkit_roots", return_value=("/usr/local/Ascend/latest",))
    @patch("gac.targets.ascend.probe._python_runtime_probe", return_value=_runtime("Ascend 910"))
    @patch("gac.targets.ascend.probe._command_probe")
    @patch("gac.targets.ascend.probe.platform.system", return_value="Linux")
    def test_plain_910_name_requires_exact_sku_confirmation(
        self, _system, command_probe, _runtime_probe, _roots
    ):
        command_probe.side_effect = lambda name, arguments: {
            "available": True,
            "ok": True,
            "executable": name,
            "returncode": 0,
            "stdout": "Name: Ascend 910" if name == "npu-smi" else "ATC version 9",
            "stderr": "",
        }
        report = probe_ascend_environment(environ={})
        self.assertTrue(report["readiness"]["torch_npu_discoverable"])
        self.assertFalse(report["readiness"]["triton_ascend_discoverable"])
        self.assertTrue(report["hardware"]["exact_sku_confirmation_required"])
        self.assertIn(
            "exact Triton-Ascend-supported product SKU is not confirmed",
            report["blockers"],
        )

    def test_cli_requirement_is_fail_closed(self):
        report = {
            "readiness": {
                "torch_npu_discoverable": False,
                "triton_ascend_discoverable": False,
            }
        }
        output = io.StringIO()
        with patch("gac.tools.probe_ascend.probe_ascend_environment", return_value=report):
            with redirect_stdout(output):
                self.assertEqual(main(["--require-torch-npu"]), 2)
        self.assertEqual(json.loads(output.getvalue()), report)


if __name__ == "__main__":
    unittest.main()
