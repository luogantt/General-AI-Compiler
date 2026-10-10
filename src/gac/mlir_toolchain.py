"""Small, explicit wrapper around the external ``mlir-opt`` executable.

IR exporters deliberately do not import or call this module.  Tool discovery and
subprocess execution only happen when a caller explicitly asks to verify MLIR.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import subprocess
import tempfile

from .errors import BackendError


DEFAULT_TIMEOUT_SECONDS = 30.0
MAX_STDERR_CHARS = 8_192
_GIT_REVISION_PATTERN = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")


class MlirToolchainError(BackendError):
    """The external MLIR toolchain could not be located or invoked."""


class MlirOptNotFoundError(MlirToolchainError):
    """No usable ``mlir-opt`` executable was found."""


class MlirVerificationError(MlirToolchainError):
    """``mlir-opt`` rejected a module or produced no verified output."""


def _binary_names() -> tuple[str, ...]:
    if os.name == "nt":
        return ("mlir-opt.exe", "mlir-opt")
    return ("mlir-opt", "mlir-opt.exe")


def _usable_file(path: Path) -> bool:
    if not path.is_file():
        return False
    return os.name == "nt" or os.access(path, os.X_OK)


def _resolve_configured_executable(
    configured: str | os.PathLike[str],
    *,
    search_path: str,
) -> Path | None:
    text = os.fspath(configured).strip()
    if not text:
        return None

    candidate = Path(text).expanduser()
    if _usable_file(candidate):
        return candidate.resolve()

    # A bare program name is useful for command-line flags such as
    # ``--mlir-opt mlir-opt``.  Values containing a directory component are
    # treated as paths, so a typo cannot silently select a different binary.
    if candidate.name == text:
        resolved = shutil.which(text, path=search_path)
        if resolved is not None and _usable_file(Path(resolved)):
            return Path(resolved).resolve()
    return None


def discover_mlir_opt(
    explicit: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    llvm_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Find ``mlir-opt`` using a deterministic, user-controllable order.

    The order is an explicit argument, ``GAC_MLIR_OPT``, ``PATH``, then
    ``build/bin`` and ``build-mlir/bin`` under ``LLVM_ROOT``.  A configured
    explicit argument or ``GAC_MLIR_OPT`` value is authoritative: when it is
    invalid this function raises immediately instead of falling through to a
    different toolchain.
    """

    environment = os.environ if env is None else env
    search_path = environment.get("PATH", os.defpath)

    if explicit is not None:
        resolved = _resolve_configured_executable(explicit, search_path=search_path)
        if resolved is None:
            raise MlirOptNotFoundError(
                f"explicit mlir-opt executable is not a usable file: {os.fspath(explicit)!r}"
            )
        return resolved

    configured = environment.get("GAC_MLIR_OPT")
    if configured is not None:
        resolved = _resolve_configured_executable(configured, search_path=search_path)
        if resolved is None:
            raise MlirOptNotFoundError(
                f"GAC_MLIR_OPT does not name a usable executable: {configured!r}"
            )
        return resolved

    path_result = shutil.which("mlir-opt", path=search_path)
    if path_result is not None and _usable_file(Path(path_result)):
        return Path(path_result).resolve()

    root_value = llvm_root if llvm_root is not None else environment.get("LLVM_ROOT")
    if root_value:
        root = Path(root_value).expanduser()
        for directory in (root / "build" / "bin", root / "build-mlir" / "bin"):
            for name in _binary_names():
                candidate = directory / name
                if _usable_file(candidate):
                    return candidate.resolve()

    raise MlirOptNotFoundError(
        "mlir-opt was not found; pass an explicit path, set GAC_MLIR_OPT, "
        "add it to PATH, or set LLVM_ROOT to a tree with build/bin or build-mlir/bin"
    )


def _normalize_command(command: Sequence[str | os.PathLike[str]]) -> tuple[str, ...]:
    if isinstance(command, (str, bytes, os.PathLike)):
        values = (os.fspath(command),)
    else:
        values = tuple(os.fspath(part) for part in command)
    if not values or any(not value for value in values):
        raise ValueError("command must contain at least one non-empty argv item")
    return values


def _validate_timeout(timeout: float) -> float:
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
        raise ValueError("timeout must be a positive finite number")
    timeout_value = float(timeout)
    if not math.isfinite(timeout_value) or timeout_value <= 0:
        raise ValueError("timeout must be a positive finite number")
    return timeout_value


def _wrap_tool_command_for_wsl(
    command: Sequence[str],
    *,
    use_wsl: bool,
    wsl_distribution: str | None,
) -> list[str]:
    """Build an argv-only local or WSL invocation for one MLIR tool."""

    if wsl_distribution is not None and not wsl_distribution:
        raise ValueError("wsl_distribution must be non-empty when provided")
    if not use_wsl:
        if wsl_distribution is not None:
            raise ValueError("wsl_distribution requires use_wsl=True")
        return list(command)
    if os.name != "nt":
        raise MlirToolchainError("use_wsl is only supported by a Windows host")

    prefix = ["wsl.exe"]
    if wsl_distribution is not None:
        prefix.extend(("-d", wsl_distribution))
    prefix.append("--")
    return prefix + list(command)


def _path_for_wsl(path: Path) -> str:
    """Map a local Windows temporary path to its default WSL mount path."""

    resolved = path.resolve()
    drive = resolved.drive
    if not drive or len(drive) != 2 or drive[1] != ":":
        raise MlirToolchainError(
            f"cannot map temporary path {str(resolved)!r} into WSL; a drive-letter path is required"
        )
    relative_parts = resolved.parts[1:]
    suffix = "/".join(part.replace("\\", "/") for part in relative_parts)
    return f"/mnt/{drive[0].lower()}/{suffix}"


def _bounded_stderr(value: str | bytes | None) -> str:
    if value is None:
        return "<no stderr>"
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    if not text:
        return "<no stderr>"
    if len(text) <= MAX_STDERR_CHARS:
        return text
    half = (MAX_STDERR_CHARS - len("\n... stderr truncated ...\n")) // 2
    return f"{text[:half]}\n... stderr truncated ...\n{text[-half:]}"


def _normalized_version_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    # Keep the tool's multi-line identity intact while making reports stable
    # across Windows and Linux hosts and removing terminal padding.
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(line.rstrip() for line in lines).strip()


def _validated_git_revision(value: str | bytes | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = value
    candidate = text.strip()
    if _GIT_REVISION_PATTERN.fullmatch(candidate) is None:
        return None
    return candidate.lower()


def probe_mlir_opt_source_revision(
    mlir_opt: str | os.PathLike[str],
    *,
    use_wsl: bool = False,
    wsl_distribution: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Read the optional build revision marker adjacent to ``mlir-opt``.

    Builds made by ``tools/build_mlir_wsl.sh`` place
    ``.gac-build-revision`` in the build root above ``bin``.  Missing,
    unreadable, or malformed markers return ``None``: a checkout revision is
    never substituted because it may not describe the selected executable.
    """

    timeout_value = _validate_timeout(timeout)
    executable_text = os.fspath(mlir_opt)
    if not executable_text:
        raise ValueError("mlir_opt must be a non-empty path")

    if not use_wsl:
        # Validate the distribution combination through the shared wrapper,
        # even though a local marker needs no subprocess.
        _wrap_tool_command_for_wsl(
            (), use_wsl=False, wsl_distribution=wsl_distribution
        )
        executable = Path(executable_text).expanduser()
        if executable.parent.name.lower() != "bin":
            return None
        marker = executable.parent.parent / ".gac-build-revision"
        try:
            contents = marker.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        return _validated_git_revision(contents)

    executable = PurePosixPath(executable_text)
    if not executable.is_absolute() or executable.parent.name != "bin":
        return None
    marker_text = str(executable.parent.parent / ".gac-build-revision")
    argv = _wrap_tool_command_for_wsl(
        ("cat", marker_text),
        use_wsl=True,
        wsl_distribution=wsl_distribution,
    )
    run_environment = None if env is None else dict(env)
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_value,
            env=run_environment,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return _validated_git_revision(completed.stdout)


def probe_mlir_opt_version(
    *,
    mlir_opt: str | os.PathLike[str] | None = None,
    command: Sequence[str | os.PathLike[str]] | None = None,
    use_wsl: bool = False,
    wsl_distribution: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
    llvm_root: str | os.PathLike[str] | None = None,
) -> str:
    """Return normalized ``mlir-opt --version`` output.

    The probe uses an argv list with ``shell=False`` and the same WSL wrapping
    and UTF-8 replacement policy as :func:`verify_mlir`.  It intentionally
    returns the tool's complete version banner instead of guessing a source
    revision that the binary did not report.
    """

    if command is not None and mlir_opt is not None:
        raise ValueError("command and mlir_opt are mutually exclusive")
    timeout_value = _validate_timeout(timeout)

    if command is None:
        executable = discover_mlir_opt(mlir_opt, env=env, llvm_root=llvm_root)
        base_command = (str(executable),)
    else:
        base_command = _normalize_command(command)
    argv = _wrap_tool_command_for_wsl(
        base_command,
        use_wsl=use_wsl,
        wsl_distribution=wsl_distribution,
    )
    argv.append("--version")

    run_environment = None if env is None else dict(env)
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_value,
            env=run_environment,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        diagnostic = _bounded_stderr(exc.stderr)
        raise MlirToolchainError(
            f"mlir-opt --version timed out after {timeout_value:g} seconds; "
            f"stderr:\n{diagnostic}"
        ) from exc
    except OSError as exc:
        raise MlirToolchainError(f"failed to launch mlir-opt --version: {exc}") from exc

    if completed.returncode != 0:
        diagnostic_value = completed.stderr or completed.stdout
        diagnostic = _bounded_stderr(diagnostic_value)
        raise MlirToolchainError(
            f"mlir-opt --version exited with status {completed.returncode}; "
            f"output:\n{diagnostic}"
        )

    version = _normalized_version_output(completed.stdout)
    if not version:
        # A few command-line tools write their version banner to stderr.  This
        # fallback preserves compatibility without treating an unrelated
        # warning as part of a normal stdout banner.
        version = _normalized_version_output(completed.stderr)
    if not version:
        raise MlirToolchainError(
            "mlir-opt --version produced no version text; verify that the selected "
            "executable is a working MLIR tool"
        )
    return version


def verify_mlir(
    source: str,
    *,
    mlir_opt: str | os.PathLike[str] | None = None,
    command: Sequence[str | os.PathLike[str]] | None = None,
    use_wsl: bool = False,
    wsl_distribution: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    env: Mapping[str, str] | None = None,
    llvm_root: str | os.PathLike[str] | None = None,
) -> str:
    """Verify and normalize an MLIR module with ``mlir-opt``.

    ``command`` allows an already selected tool invocation, for example
    ``("/home/user/build/bin/mlir-opt",)`` with ``use_wsl=True``.  Supplying a
    command bypasses discovery.  Otherwise ``mlir_opt`` and the documented
    discovery order are used.  The returned text is the canonicalized/CSE'd
    module emitted by the external tool.
    """

    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if command is not None and mlir_opt is not None:
        raise ValueError("command and mlir_opt are mutually exclusive")
    timeout_value = _validate_timeout(timeout)

    if command is None:
        executable = discover_mlir_opt(mlir_opt, env=env, llvm_root=llvm_root)
        base_command = (str(executable),)
    else:
        base_command = _normalize_command(command)

    run_environment = None if env is None else dict(env)
    with tempfile.TemporaryDirectory(prefix="gac-mlir-") as directory:
        root = Path(directory)
        input_path = root / "input.mlir"
        output_path = root / "verified.mlir"
        input_path.write_text(source, encoding="utf-8", newline="\n")

        input_argument = str(input_path)
        output_argument = str(output_path)
        if use_wsl:
            input_argument = _path_for_wsl(input_path)
            output_argument = _path_for_wsl(output_path)
        argv = _wrap_tool_command_for_wsl(
            base_command,
            use_wsl=use_wsl,
            wsl_distribution=wsl_distribution,
        )

        argv.extend(
            (
                "--verify-each",
                "--canonicalize",
                "--cse",
                input_argument,
                "-o",
                output_argument,
            )
        )

        try:
            completed = subprocess.run(
                argv,
                shell=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_value,
                env=run_environment,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            diagnostic = _bounded_stderr(exc.stderr)
            raise MlirVerificationError(
                f"mlir-opt timed out after {timeout_value:g} seconds; stderr:\n{diagnostic}"
            ) from exc
        except OSError as exc:
            raise MlirToolchainError(f"failed to launch mlir-opt: {exc}") from exc

        if completed.returncode != 0:
            diagnostic = _bounded_stderr(completed.stderr)
            raise MlirVerificationError(
                f"mlir-opt exited with status {completed.returncode}; stderr:\n{diagnostic}"
            )

        try:
            verified = output_path.read_text(encoding="utf-8")
        except OSError as exc:
            diagnostic = _bounded_stderr(completed.stderr)
            raise MlirVerificationError(
                f"mlir-opt did not create a readable output file; stderr:\n{diagnostic}"
            ) from exc
        if not verified.strip():
            diagnostic = _bounded_stderr(completed.stderr)
            raise MlirVerificationError(
                f"mlir-opt produced empty output; stderr:\n{diagnostic}"
            )
        return verified
