# MSDA 自定义插件实验流程

默认 `adapt`、构建和运行流程仍使用 NVIDIA 官方 MSDA。只有显式运行 `msda-rewrite`，并在构建、加载时传入 `--plugin-library`，才启用实验插件 `GACMSDeformAttn` v1。插件面向 TensorRT 10.13；混合 INT8 校准范围工具进一步锁定 **10.13.3.9**。

本页说明复现方法。实测数据单独记录在 [RTX 3070 验证报告](validation-msda-3070.md)，以报告中的库 SHA、输入 SHA、GPU、版本和测试范围为准。快速筛选或单个算子的结果不能直接作为整模型加速结论。

## 1. 构建插件

以下命令使用 Bash，从 `standalone/bevformer-tensorrt` 执行。先准备原始完整 `adapted.onnx`、与其绑定的真实校准 `ranges.json`、符合[输入契约](model-and-data.md)的真实 `images.npy`。每次实验使用新目录。

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export NVIDIA_TF32_OVERRIDE=0 CUDA_MODULE_LOADING=LAZY
MODEL="$PWD/models/bevformer-base/adapted.onnx"
RANGES="$PWD/models/bevformer-base/ranges.json"
IMAGES="$PWD/results/real-images.npy"
RUN="$PWD/results/msda-experiment-01"
mkdir "$RUN"

cmake -S native/msda -B build/msda-experiment-01 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DTENSORRT_INCLUDE_DIR=/path/to/TensorRT-10.13/include \
  -DTENSORRT_LIBRARY_DIR=/path/to/site-packages/tensorrt_libs
cmake --build build/msda-experiment-01 --parallel
LIB="$PWD/build/msda-experiment-01/libgac_msda_plugin.so"
```

pip 环境可能只有运行库，没有头文件；也可通过 `TENSORRT_LIBRARY` 指定完整的 `libnvinfer.so.10` 路径。ABI、序列化字段、Windows 构建和 kernel C 接口见 [native/msda/README.md](../native/msda/README.md)。

优化保留五输入、INT32 level 元数据以及 FP32/FP16 同型输出；当前只接受非空静态形状。`channels_per_thread` 为 1/2/4，`block_threads` 为 128/256/512/768，`specialize` 为 0 通用、1 自动、2 强制 `L=4,P=8,C=32`。`prefetch=1` 只提前加载同一 level 下一采样点的坐标和 attention 权重，**没有 Value 预取或共享内存流水线**。

FP16 使用显式 half FMA，固定坐标变换、双线性插值配对和逐点累加的舍入顺序，对齐已测试的 NVIDIA TensorRT 10.13.3 / SM86 实现。`--fmad=false` 限制意外融合，显式 `__hfma` / `__hfma2` 仍然融合；累加仍为 FP16。逐位相等是需要测试确认的结果，不能从公式相同或版本号相同直接推出。

## 2. 从真实完整模型捕获 MSDA 边界

默认选择第一层 Encoder Spatial Cross-Attention。构建和执行分进程，构建阶段不导入 PyTorch，减少 8 GB GPU 上的额外显存占用。不要同时运行其他构建或计时实验。

```bash
python scripts/capture_msda.py \
  --onnx "$MODEL" --images "$IMAGES" \
  --precision mixed-int8 --ranges "$RANGES" \
  --node-substring encoder/layers.0/attentions.1 \
  --output "$RUN/capture-build" --build-only

python scripts/capture_msda.py \
  --engine "$RUN/capture-build/capture.engine" \
  --images "$IMAGES" --output "$RUN/capture-run"
```

脚本先解析完整原图，使用原 ONNX SHA 应用完整精度策略和校准范围，再取消原输出标记并暴露选中 MSDA 的五输入及官方输出。FP16 边界通过 Identity 和输出类型约束固定，构建后检查实际 engine I/O dtype。`--precision fp32` 或 `fp16` 可用于浮点对照，此时不传 `--ranges`。

输出 `capture.npz` 包含 `value/shapes/starts/locations/weights/expected`；旁边的 `capture.report.json` 记录节点、形状、dtype、来源 SHA、有限性与 level 越界检查。`expected` 来自该捕获引擎中的官方 MSDA。增加输出及移除下游输出可能改变 tactic、reformat 和图裁剪，因此不能声称捕获结果逐位等同于原完整引擎的内部激活。

## 3. 正确性检查、搜索和独立复测

```bash
python scripts/benchmark_msda.py --library "$LIB" \
  --checks-only --report "$RUN/kernel-checks.json"

python scripts/benchmark_msda.py --library "$LIB" \
  --inputs "$RUN/capture-run/capture.npz" --dtype fp16 \
  --warmup 10 --repeat 50 --report "$RUN/tune-fp16.json"
```

完整参数网格每个 dtype 有 48 组；加 `--quick` 是 18 组初筛。脚本先跑小型 CPU oracle、NVIDIA 插件和自定义插件对照，再对相同真实输入进行候选筛选；FP16 候选要求逐位一致。计时使用单算子 CUDA Graph replay，并交替测量官方基线与候选，不能换算为整模型 FPS。

检查报告中 `inputs.floating_cast=false`，以及 `captured_reference_comparison.exact=true`，确保官方单算子复现了保存的官方输出。只使用通过数值检查的候选。提取搜索结果后，在独立进程复测：

```bash
python - "$RUN" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
report = json.loads((root / "tune-fp16.json").read_text())
assert report["status"] == "measured"
assert report["inputs"]["floating_cast"] is False
assert report["captured_reference_comparison"]["exact"] is True
with (root / "selection.json").open("x") as target:
    json.dump(report["selection"], target, indent=2)
PY

python scripts/benchmark_msda.py --library "$LIB" \
  --inputs "$RUN/capture-run/capture.npz" --dtype fp16 \
  --config "$RUN/selection.json" --warmup 10 --repeat 100 \
  --report "$RUN/retest-fp16.json"
```

搜索最优值存在选择偏差；先确认复测数值与速度稳定，再接入模型。更换 GPU、TensorRT 二进制、dtype 或输入形状后重新验证。更换 `--node-substring` 可捕获其他层；第一层的结果不能证明其余五层同样准确或同样快。

## 4. 只替换六个 Spatial MSDA

Base 共 18 个 MSDA：6 个 Temporal、6 个 Spatial、6 个 Decoder。下面显式选择 Encoder 的六个 `attentions.1/deformable_attention`，将复测配置用于这六层作为整图实验候选。`--node` 接受**完整节点名**；省略会替换全部 MSDA。

```bash
python - "$RUN" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
report = json.loads((root / "retest-fp16.json").read_text())
assert report["status"] == "measured"
names = [f"/pts_bbox_head/transformer/encoder/layers.{i}/attentions.1/"
         "deformable_attention/MSDeformAttn" for i in range(6)]
with (root / "spatial6-config.json").open("x") as target:
    json.dump({name: report["selection"] for name in names}, target, indent=2)
PY

MSDA_NODES=()
for i in {0..5}; do
  MSDA_NODES+=(--node "/pts_bbox_head/transformer/encoder/layers.${i}/attentions.1/deformable_attention/MSDeformAttn")
done
python -m bevformer_trt msda-rewrite \
  --onnx "$MODEL" --output "$RUN/spatial6.onnx" \
  --report "$RUN/rewrite.json" --expected-msda-count 18 \
  "${MSDA_NODES[@]}" --node-config "$RUN/spatial6-config.json"
```

检查 `rewrite.json` 的 `replaced_node_count=6`。工具保留其他节点、权重、边、形状与 tensor 名称，并记录每个替换节点的参数；结构证明不等于数值证明。若各层独立调优结果不同，可以在 `spatial6-config.json` 中分别设置。

## 5. 显式重新绑定原校准范围

改写 ONNX 后模型 SHA 已改变，不能直接把原 `ranges.json` 用于新模型，也不要手工修改 hash。

```bash
python -m bevformer_trt msda-rebind-ranges \
  --source-onnx "$MODEL" --onnx "$RUN/spatial6.onnx" \
  --ranges "$RANGES" --output "$RUN/spatial6-ranges.json"
```

该步骤证明模型只改了声明的 MSDA，且被校准的 backbone tensor 没有 MSDA 上游依赖，再生成带派生证据的新绑定。原测量 cache、范围和来源记录保留；它没有重新校准，也没有证明自定义 MSDA 精度。校准策略见[混合精度与校准](precision-and-calibration.md)。

## 6. 构建并比较完整引擎

先用当前工具重建官方基线并导出 timing cache，再让自定义构建复用该缓存。两次构建采用相同精度、workspace、tactic DRAM、优化级别和辅助流设置，运行使用同一真实输入。自定义引擎的构建与运行都必须加载同一插件库。

```bash
python -m bevformer_trt build --onnx "$MODEL" \
  --precision mixed-int8 --calibrated-ranges "$RANGES" \
  --workspace-mib 1024 --builder-optimization-level 3 \
  --tactic-dram-mib 4096 --max-aux-streams 0 \
  --timing-cache-output "$RUN/stock.timing.cache" \
  --engine "$RUN/stock.engine" --report "$RUN/stock-build.json"

python -m bevformer_trt build --onnx "$RUN/spatial6.onnx" \
  --precision mixed-int8 --calibrated-ranges "$RUN/spatial6-ranges.json" \
  --workspace-mib 1024 --builder-optimization-level 3 \
  --tactic-dram-mib 4096 --max-aux-streams 0 \
  --timing-cache-input "$RUN/stock.timing.cache" \
  --timing-cache-output "$RUN/spatial6.timing.cache" \
  --plugin-library "$LIB" \
  --engine "$RUN/spatial6.engine" --report "$RUN/spatial6-build.json"

python -m bevformer_trt run --engine "$RUN/stock.engine" \
  --inputs "$IMAGES" --warmup 5 --repeat 30 --output "$RUN/stock-run"

python -m bevformer_trt run --engine "$RUN/spatial6.engine" \
  --plugin-library "$LIB" --inputs "$IMAGES" \
  --reference "$RUN/stock-run/outputs.npz" --rtol 1e-3 --atol 1e-4 \
  --warmup 5 --repeat 30 --output "$RUN/spatial6-run"
```

Timing cache 保存并复用 TensorRT 的 tactic 测量数据，**不是 INT8 calibration cache**，不能替代 `ranges.json` 或其模型绑定。缓存复用不保证两次构建最终选择相同 tactic，更不保证输出逐位相同。输出缓存使用新文件，保留官方基线缓存；仍须比较两份 build 的 `.layers.json`，检查上游计算、融合、reformat 与插件实际精度。

共享 timing cache 的受控构建已完成：本次同一真实输入上，自定义引擎与新建官方基线的三个输出逐位一致，新建官方基线与原官方引擎也逐位一致。层检查中全部 18 个 MSDA 的输入输出描述相同，上游六个 Spatial value projection 的融合与 tactic 相同；全图暴露的 tactic、shape、dtype 和 layout 未发现差异。受控 A/B/B/A 整图测量、来源 SHA、初次构建差异及适用范围见 [RTX 3070 验证报告](validation-msda-3070.md)。这一结论只覆盖报告中的输入和环境，不能替代数据集 mAP/NDS 验证。

检查 run 的逐输出误差报告，覆盖 `bev_embed`、分类分数与框预测。示例容差须结合验证目标事先确定；失败时调查差异，不能通过不断放宽容差宣布通过。**输出有限仅排除了 NaN/Inf，不证明数值正确或检测精度不变。**

整图时间来自独立的完整推理测量；输入复制、初始化和预处理不计入该时间。需要在相同温度、功耗与负载条件下交替复测，避免把单算子 CUDA Graph 收益或不同运行条件混为整图收益。逐位一致、容差内一致和检测任务指标是不同层次的证据；本流程本身不提供 nuScenes **mAP/NDS** 不下降的证明。

## 7. 继续优化：与已经优化的配置比较

新候选必须直接比较上一轮赢家，不能继续只与较慢的官方实现比较。`--baseline-config` 只改变性能基线；数值仍以 NVIDIA 官方输出为准。`--configs` 接受完整参数对象的 JSON 数组；它与单配置复测 `--config` 互斥。

```bash
cat > "$RUN/previous.json" <<'JSON'
{"channels_per_thread":4,"block_threads":128,"specialize":1,"prefetch":0}
JSON
cat > "$RUN/index32.json" <<'JSON'
{"channels_per_thread":4,"block_threads":128,"specialize":3,"prefetch":0}
JSON
python scripts/benchmark_msda.py --library "$LIB" \
  --inputs "$RUN/capture.npz" --baseline-config "$RUN/previous.json" \
  --config "$RUN/index32.json" --warmup 10 --repeat 60 \
  --report "$RUN/retest-fp16.json"
```

`specialize=3` 在 `L=4/P=8/C=32` 且所有相关张量字节跨度不超过 `INT32_MAX` 时使用 32 位索引，同时复用相邻像素的步长；其他情况自动回退 64 位。`prefetch=2` 是真实四邻点 Value 预取，属于待测候选，**不是默认开启的优化**。参数支持并不代表所有形状都获益。

使用新的 `$RUN` 目录保存这次复测，再按第 4～6 步改写六层、绑定范围和构建。当前源码在第二轮整网验证中选择了上面的 `index32.json` 配置，复测延时降低 2.60%；中途发现并修复的重构回归也保留在[第二轮报告](validation-msda-round2-3070.md)，不能用新库内的两个参数对照替代与旧库的实际比较。

重建插件时保留旧 `.so`、engine 和报告；比较完整模型时分别启动进程，加载对应版本的库。不能在同一进程注册两个同名 TensorRT plugin creator 来代表两个版本。

## 8. 找瓶颈与验证 CUDA Graph

```bash
# 根据 TensorRT Inspector 精确匹配执行层类型；融合层保持完整。
python scripts/summarize_profile.py \
  --profile-dir results/profile --output-dir results/profile-summary

# 完整引擎 Graph 探测：失败会保存原因，不降级成部分图或报告虚假加速。
python scripts/benchmark_engine_graph.py --engine "$RUN/spatial6.engine" \
  --inputs "$IMAGES" --plugin-library "$LIB" \
  --output-dir "$RUN/graph-probe" --warmup 5 --repeat 20

# 先完成独立数值测试，再采硬件指标；需要主机授予性能计数器访问权限。
ncu --set basic --kernel-name regex:msda_kernel --launch-skip 2 --launch-count 1 \
  --export "$RUN/msda-counters" python scripts/profile_msda_kernel.py \
  --library "$LIB" --inputs "$RUN/capture.npz" --config "$RUN/index32.json"
```

逐层回调占比不是无重叠的 GPU 时间分解。Nsight replay 会扰动运行，用于分析硬件行为，不能替代无 profiler 的整网计时。`ERR_NVGPUCTRPERM` 表示计数器权限不足；此时不能声称已测得带宽、occupancy 或访存等待比例。
