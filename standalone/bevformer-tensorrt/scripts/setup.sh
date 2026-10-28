#!/usr/bin/env bash
# Install the standalone package without changing the host CUDA driver.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="python3.11"
venv_dir="${project_dir}/.venv"
mirror="pypi"
reuse_system=0
install_torch=0

usage() {
  cat <<'EOF'
Usage: bash scripts/setup.sh [options]
  --python PATH           Python 3.11 executable (default: python3.11)
  --venv PATH             Virtual environment path (default: project/.venv)
  --mirror pypi|tsinghua|ustc
                          Index for CPU dependencies and package installation
  --isolated              Create isolated venv (default)
  --reuse-system-packages Make base Python site packages visible to a new venv
  --install-torch         Explicitly install Torch 2.7.0 from official cu126 index
  --help                  Show help

New environments are isolated by default. Use --install-torch for a fresh setup.
--reuse-system-packages exposes only the base Python installation's packages;
it does not inherit a different venv's Torch. To reuse an existing venv, activate
it and install this package directly as documented in README.md.
TensorRT is pinned to 10.13.3.9 and installed from PyPI. Drivers are not modified.
EOF
}

while (($#)); do
  case "$1" in
    --python|--venv|--mirror)
      if (($# < 2)); then
        printf 'Missing value for %s\n' "$1" >&2
        exit 2
      fi
      case "$1" in
        --python) python_bin="$2" ;;
        --venv) venv_dir="$2" ;;
        --mirror) mirror="$2" ;;
      esac
      shift 2 ;;
    --isolated) reuse_system=0; shift ;;
    --reuse-system-packages) reuse_system=1; shift ;;
    --install-torch) install_torch=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage; exit 2 ;;
  esac
done

case "$mirror" in
  pypi) index_url="https://pypi.org/simple" ;;
  tsinghua) index_url="https://pypi.tuna.tsinghua.edu.cn/simple" ;;
  ustc) index_url="https://mirrors.ustc.edu.cn/pypi/simple" ;;
  *) printf 'Unsupported mirror: %s\n' "$mirror" >&2; exit 2 ;;
esac

"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use Python 3.11 for the recorded environment"'
if [[ ! -x "${venv_dir}/bin/python" ]]; then
  if [[ "$reuse_system" == 1 ]]; then
    "$python_bin" -m venv --system-site-packages "$venv_dir"
  else
    "$python_bin" -m venv "$venv_dir"
  fi
fi
venv_python="${venv_dir}/bin/python"

if [[ "$install_torch" == 1 ]]; then
  "$venv_python" -m pip install -r "${project_dir}/requirements-torch-cu126.txt"
fi
# Fail early rather than silently downloading a multi-GB Torch installation.
"$venv_python" - <<'PY'
try:
    import torch
except ImportError as error:
    raise SystemExit(
        "CUDA Torch is required. Rerun with --install-torch, or install into your existing CUDA venv."
    ) from error
if torch.version.cuda is None:
    raise SystemExit("CPU-only Torch found; rerun with --isolated --install-torch in a new venv.")
print(f"Using Torch {torch.__version__}; CUDA runtime {torch.version.cuda}")
PY

"$venv_python" -m pip install --index-url "$index_url" -r "${project_dir}/requirements.txt"
"$venv_python" -m pip install --index-url https://pypi.org/simple -r "${project_dir}/requirements-tensorrt.txt"
"$venv_python" -m pip install --index-url "$index_url" --no-deps -e "$project_dir"
"$venv_python" -m bevformer_trt --help
printf '\nInstalled. Activate with:\n  source %q/bin/activate\n' "$venv_dir"
printf 'Then run: python -m bevformer_trt check --report results/environment.json\n'
