from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "galaxeavla" / "apply_ascend_overlay.py"
PATCH = (
    ROOT
    / "integrations"
    / "galaxeavla"
    / "patches"
    / "0001-ascend-torchnpu-runtime.patch"
)
SPEC = importlib.util.spec_from_file_location("apply_ascend_overlay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _run_git(repo: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _find_upstream_checkout() -> Path | None:
    candidates = []
    configured = os.environ.get("GALAXEAVLA_UPSTREAM")
    if configured:
        candidates.append(Path(configured))
    candidates.append(ROOT.parent / "GalaxeaVLA-upstream")
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        result = _run_git(candidate, "rev-parse", "HEAD", check=False)
        if (
            result.returncode == 0
            and result.stdout.strip().lower() == MODULE.EXPECTED_UPSTREAM_REVISION
        ):
            return candidate.resolve()
    return None


class GalaxeaVLAAscendOverlayTests(unittest.TestCase):
    def test_patch_asset_is_pinned_and_contains_no_local_secret_paths(self):
        payload = PATCH.read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), MODULE.EXPECTED_PATCH_SHA256)
        text = payload.decode("utf-8")
        self.assertIn("src/g05/utils/common/device.py", text)
        self.assertIn("tests/test_device_utils.py", text)
        self.assertIn("tests/test_qwen35_device_fallbacks.py", text)
        self.assertNotRegex(text, r"[A-Za-z]:\\")
        self.assertNotIn("/root/", text)
        self.assertNotRegex(text, r"hf_[A-Za-z0-9]+")

    def test_patch_checks_and_applies_in_a_clean_temporary_worktree(self):
        source = _find_upstream_checkout()
        if source is None:
            self.skipTest(
                "set GALAXEAVLA_UPSTREAM to an OpenGalaxea/GalaxeaVLA checkout at "
                f"{MODULE.EXPECTED_UPSTREAM_REVISION}"
            )

        with tempfile.TemporaryDirectory(prefix="g05-overlay-test-") as temporary:
            worktree = Path(temporary) / "upstream"
            _run_git(
                source,
                "worktree",
                "add",
                "--detach",
                str(worktree),
                MODULE.EXPECTED_UPSTREAM_REVISION,
            )
            try:
                stdout = io.StringIO()
                stderr = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    check_result = MODULE.main([str(worktree)])
                self.assertEqual(check_result, 0, stderr.getvalue())
                self.assertIn("no files were changed", stdout.getvalue())
                self.assertEqual(MODULE.worktree_changes(worktree), [])

                stdout = io.StringIO()
                stderr = io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    apply_result = MODULE.main([str(worktree), "--apply"])
                self.assertEqual(apply_result, 0, stderr.getvalue())
                self.assertIn("Applied TorchNPU overlay", stdout.getvalue())
                self.assertEqual(
                    _run_git(worktree, "rev-parse", "HEAD").stdout.strip().lower(),
                    MODULE.EXPECTED_UPSTREAM_REVISION,
                )
                self.assertTrue(MODULE.worktree_changes(worktree))
                _run_git(worktree, "diff", "--check")

                static_files = (
                    worktree / "src" / "g05" / "utils" / "common" / "device.py",
                    worktree / "tests" / "test_device_utils.py",
                    worktree / "tests" / "test_qwen35_device_fallbacks.py",
                )
                for source_file in static_files:
                    self.assertTrue(source_file.is_file(), source_file)
                    ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))

                inferencer_source = (
                    worktree / "src" / "g05" / "models" / "g05" / "inferencer.py"
                ).read_text(encoding="utf-8")
                policy_source = (
                    worktree / "src" / "g05" / "models" / "g05" / "g05_policy.py"
                ).read_text(encoding="utf-8")
                vision_source = (
                    worktree / "src" / "g05" / "models" / "g05" / "qwen35" / "vision.py"
                ).read_text(encoding="utf-8")
                gdn_source = (
                    worktree
                    / "src"
                    / "g05"
                    / "models"
                    / "g05"
                    / "qwen35"
                    / "gated_deltanet.py"
                ).read_text(encoding="utf-8")
                self.assertNotIn('torch.autocast("cuda"', inferencer_source)
                self.assertNotIn("torch.cuda.synchronize", policy_source)
                self.assertIn('query_states.device.type == "cuda"', vision_source)
                self.assertIn("else torch_chunk_gated_delta_rule", gdn_source)

                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                    io.StringIO()
                ):
                    second_apply_result = MODULE.main([str(worktree), "--apply"])
                self.assertEqual(second_apply_result, 2)
            finally:
                _run_git(source, "worktree", "remove", "--force", str(worktree))
                _run_git(source, "worktree", "prune")


if __name__ == "__main__":
    unittest.main()
