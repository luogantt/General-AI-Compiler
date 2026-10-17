set -eu
cd /root/gac-tensorrt-20261017
export NVIDIA_TF32_OVERRIDE=0
export CUDA_MODULE_LOADING=LAZY
venv/bin/python tools/tensorrt/profile_engine.py --engine results/mixed-01/full.engine --inputs models/held-out-images.npy --output-dir results/profile-20261017/mixed-layers --mode layers --warmup 10 --repeat 30 > results/profile-20261017/mixed-layers.log 2>&1
cat results/profile-20261017/mixed-layers.log
venv/bin/python tools/tensorrt/profile_engine.py --engine results/fp32-workspace-01/full.engine --inputs models/held-out-images.npy --output-dir results/profile-20261017/fp32-layers --mode layers --warmup 10 --repeat 30 > results/profile-20261017/fp32-layers.log 2>&1
cat results/profile-20261017/fp32-layers.log
