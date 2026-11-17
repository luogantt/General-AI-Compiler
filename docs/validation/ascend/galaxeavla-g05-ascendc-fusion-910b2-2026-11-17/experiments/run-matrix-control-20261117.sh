#!/bin/bash
set -euo pipefail
export ASCEND_CUSTOM_OPP_PATH=/root/work/gac-qknorm-opp-v4/vendors/gacqknorm
export GAC_QKNORM_ROPE_LIBRARY=/root/work/gac-qknorm-bridge-v4/lib/gac_g05_ascendc_qknorm_rope_bridge.so
export PYTHONPATH=/root/work:/root/work/GalaxeaVLA-upstream/src:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python /root/work/benchmark_g05_libero.py \
  --source-dir /root/work/GalaxeaVLA-upstream \
  --asset-root /root/work/models/galaxeavla-g05 \
  --checkpoint /root/work/models/galaxeavla-g05/g05-libero/model.pt \
  --input-dir /root/work/libero_input \
  --output-dir /root/work/libero-bf16-native-matrix-control-20261117 \
  --device npu:0 --weights bf16 \
  --ae-backend torchair --ae-options '{"frozen_parameter":true}' \
  --vlm-backend torchair_aclgraph --vlm-options '{"frozen_parameter":true}' \
  --gdn-algorithm triton --ae-attention npu --skip-unused-action-codec \
  --precompute-time-conditioning --time-conditioning-lookup request --static-vision \
  --ae-rope matrix --ae-norm npu --functional-mrope \
  --noise /root/work/libero-bf16-eager/noise.npy \
  --timing-noise native --seed 0 --warmup 5 --iterations 20 --verify \
  --reference-report /root/work/libero-bf16-native-vision-rope-norm-mrope/report.json \
  --trace-dir /root/work/profile-libero-matrix-control-20261117
