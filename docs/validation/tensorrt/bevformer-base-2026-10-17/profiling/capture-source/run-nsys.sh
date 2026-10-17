set -eu
cd /root/gac-tensorrt-20261017
export NVIDIA_TF32_OVERRIDE=0
export CUDA_MODULE_LOADING=LAZY
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-memory-usage=true --export=sqlite --output=results/profile-20261017/mixed-nsys venv/bin/python tools/tensorrt/profile_engine.py --engine results/mixed-01/full.engine --inputs models/held-out-images.npy --output-dir results/profile-20261017/mixed-trace --mode trace --warmup 10 --repeat 5 > results/profile-20261017/mixed-nsys.log 2>&1
cat results/profile-20261017/mixed-nsys.log
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_pushpop_sum --format csv --output results/profile-20261017/mixed-stats results/profile-20261017/mixed-nsys.sqlite
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-memory-usage=true --export=sqlite --output=results/profile-20261017/fp32-nsys venv/bin/python tools/tensorrt/profile_engine.py --engine results/fp32-workspace-01/full.engine --inputs models/held-out-images.npy --output-dir results/profile-20261017/fp32-trace --mode trace --warmup 10 --repeat 5 > results/profile-20261017/fp32-nsys.log 2>&1
cat results/profile-20261017/fp32-nsys.log
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_pushpop_sum --format csv --output results/profile-20261017/fp32-stats results/profile-20261017/fp32-nsys.sqlite
