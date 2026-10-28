# DCNv2 第三轮优化：完整 BEVFormer-base 降到 309.42 ms

**本轮同条件对照：325.97 → 309.42 ms/帧，延时降低 5.07%，约 3.23 FPS。** 设备为 RTX 3070 Ti Laptop 8 GB；三个最终输出与本轮基线、原官方插件参考逐位一致。[原始汇总](validation/dcn-2026-10-22/summary.json)

这是 TensorRT 路径中的 C++/CUDA 插件优化。模型结构、权重、采样点数量、FP16 算术和 cuBLAS 矩阵乘保持不变。首轮 MSDA 的 18.05%、第二轮的 2.60% 和本轮的 5.07% 来自各自对照，**不能相加**。

## 改了什么

DCNv2 先根据学习到的 offset 和 mask 采样图像特征，展开成列矩阵，再与卷积权重相乘：

```text
输入特征 + offset + mask
          ↓ im2col：双线性采样、乘 mask
列矩阵 [Cin × 3 × 3, Hout × Wout]
          ↓ cuBLAS HGEMM，与卷积权重相乘
输出特征 [1, Cout, Hout, Wout]
```

同一 deformable group 内，不同输入通道使用相同的 offset 和 mask。原先每个线程处理一个通道、一个输出位置；新候选让每个线程处理 **4 个通道**，复用坐标、四个邻点索引及插值系数。每个通道仍读取自己的特征值，写回原来的列矩阵位置。相邻线程继续覆盖相邻空间位置，保持 NCHW 访问的连续性。

同时固定 3×3 循环，比较 128/256/512/1024 线程块。本轮采用 **block=128、specialize=4**。FP16 的坐标舍入、插值运算顺序和 mask 乘法保持一致；矩阵乘仍调用 `cublasHgemm`。4 通道 Kernel 使用 51 个寄存器，编译报告未发现 spill；这不等于硬件计数器已证明瓶颈归因。

本图共 156 个 DCNv2 节点：ResNet-101 的 26 个 DCNv2 模块分别处理六路相机。它们全部满足本次插件的窄接口：**静态 NCHW、batch=1、FP16、3×3、group=deformable_group=1、无 bias**。其他形状不会自动套用该插件。

## 先测单算子，再测整网

从同一真实六相机输入提取三组中间张量。提取工具在完整原图上先验证精度策略与校准来源，再暴露选中插件的输入和输出。暴露中间输出可能改变上游 tactic，因此这些张量属于捕获引擎，不能宣称与原整网的每个中间张量逐位一致。

以下是**独立 TensorRT 插件**的复测，包含 im2col 和 HGEMM；使用 CUDA Graph 重复 20 次，每组交替计时 40 次，捕获后清空输出并 replay 验证：

| 捕获层 | 输入尺寸 | 官方 DCNv2 | 新插件 | 单算子延时降低 |
|---|---|---:|---:|---:|
| layer3.0 / C4 | `[1,256,58,100]` | 0.41439 ms | 0.28521 ms | 31.17% |
| layer3.1 / C4 | `[1,256,58,100]` | 0.41357 ms | 0.28434 ms | 31.25% |
| layer4.1 / C5 | `[1,512,29,50]` | 0.30152 ms | 0.24555 ms | 18.56% |

三组均与 NVIDIA 插件输出逐位一致。[C4-0](validation/dcn-2026-10-22/plugin-01.json) · [C4-1](validation/dcn-2026-10-22/plugin-02.json) · [C5](validation/dcn-2026-10-22/plugin-03.json)

早期通用 im2col 候选比官方更慢；单纯 3×3 展开只有约 5%～6% 的单算子收益。上述较大收益来自后续多通道复用。保留了[早期结果](validation/dcn-2026-10-22/dcn-probe-01.json)和[1/2/4 通道搜索](validation/dcn-2026-10-22/dcn-cpt-01.json)，没有用失败候选替换最佳配置。

## 完整引擎对照

基线是第二轮已验证的 MSDA 优化引擎。本轮保留其六个 Spatial MSDA 配置与库，仅用新插件替换 156 个 DCNv2。使用同一原始 timing cache、构建等级 3、1 GiB workspace、4 GiB tactic DRAM 限制和零辅助流；冻结原模型已测量的 INT8 校准范围，不重新校准。

每次单独启动进程，预热 5 次、测量 30 次；开始前温度 ≤55°C、GPU 利用率 <10%，按 A/B/B/A 顺序执行：

| 顺序 | 配置 | 每帧中位延时 |
|---|---|---:|
| A1 | 已优化 MSDA，官方 DCNv2 | 325.028 ms |
| B1 | 已优化 MSDA，新 DCNv2 | 309.463 ms |
| B2 | 已优化 MSDA，新 DCNv2 | 309.340 ms |
| A2 | 已优化 MSDA，官方 DCNv2 | 326.574 ms |

合并每种配置的 60 个原始样本，得到 **325.966 / 309.424 ms**，加速比 **1.05346×**。GPU 未锁频，两个新版本运行均快于两个基线运行。[协议](validation/dcn-2026-10-22/protocol.json) · [输出比较](validation/dcn-2026-10-22/comparison.json) · [构建来源](validation/dcn-2026-10-22/build-summary.json)

Inspector 两侧均为 3816 层。9 个临时张量名称变化已逐一验证生产/消费关系相同；归一化后，仅 156 个 DCNv2 的插件类型和版本变化，其余 3660 层所有公开字段一致，未发现其他 tactic、类型或布局变化。Inspector 不暴露全部内部实现；63 条记录排列顺序变化不直接代表设备执行顺序变化。[逐层审计](validation/dcn-2026-10-22/tactic-audit.json)

两种引擎报告的 context 显存需求均为 2,053,694,976 B；绑定后设备显存快照也相同。该快照**不是峰值显存**，不能用它承诺部署所需显存上限。

## 正确性与范围

- 160 项本地 CPU 测试通过，覆盖配置、ONNX 改写证明、校准约束及既有功能。
- GPU：16 个形状/精度案例，320 项候选比较、42 项非法参数检查、4 项独立 FP32 数学参考检查通过。包含 batch>1、分组尾通道、边界、stride/dilation、非 3×3 回退及输出越界哨兵。[检查记录](validation/dcn-2026-10-22/dcn-kernel-checks.json)
- 同一个新插件引擎创建两个 execution context，在不同 Stream 上交替提交不同输入，10 轮、20 个输出均与各自的官方参考逐位一致。[上下文检查](validation/dcn-2026-10-22/dcn-context-check.json)
- 完整模型三个输出：`all_bbox_preds`、`all_cls_scores`、`bev_embed`，均与基线和官方参考逐位一致。

测试仍是 **BEVFormer-base、六路 `[1,6,3,928,1600]` 图像、混合 INT8/FP16、固定几何、无历史帧**。测量止于原始检测头张量，不含图像预处理、主机拷贝、首次加载和结果解码；没有新增 nuScenes mAP/NDS 验收，也未证明任意输入都逐位相同。

## 代码与复现

| 文件 | 作用 |
|---|---|
| [dcn_im2col.cu](../native/dcn/dcn_im2col.cu) | CUDA 采样与列矩阵，1/2/4 通道候选 |
| [plugin.cpp](../native/dcn/plugin.cpp) | TensorRT IPluginV3 接口、workspace、cuBLAS 资源和 HGEMM |
| [dcn.py](../src/bevformer_trt/dcn.py) | 仅替换配置指定的节点，完整 protobuf 还原验证 |
| [capture_dcn.py](../scripts/capture_dcn.py) | 提取真实中间输入及官方输出 |
| [benchmark_dcn.py](../scripts/benchmark_dcn.py) | 官方插件对照、候选搜索、Graph replay 检查 |

先按 [DCNv2 构建说明](../native/dcn/README.md)生成 `libgac_dcn_plugin.so`。基于第二轮六个 Spatial MSDA 已改写的 ONNX 与对应 ranges，使用本轮[精确节点配置](validation/dcn-2026-10-22/config.json)：

```bash
python -m bevformer_trt build \
  --onnx /path/to/round2/adapted.onnx \
  --precision mixed-int8 --calibrated-ranges /path/to/round2/ranges.json \
  --dcn-config docs/validation/dcn-2026-10-22/config.json \
  --plugin-library /path/to/libgac_msda_plugin.so \
  --plugin-library /path/to/libgac_dcn_plugin.so \
  --engine build/round3/candidate.engine --report build/round3/build.json \
  --workspace-mib 1024 --tactic-dram-mib 4096 --max-aux-streams 0 \
  --builder-optimization-level 3 \
  --timing-cache-input /path/to/original/stock.timing.cache

python -m bevformer_trt run \
  --engine build/round3/candidate.engine --inputs /path/to/real-images.npy \
  --plugin-library /path/to/libgac_msda_plugin.so \
  --plugin-library /path/to/libgac_dcn_plugin.so \
  --output build/round3/run --warmup 5 --repeat 30
```

构建报告同时记录**源 ONNX 哈希、实际解析图哈希、配置哈希及原校准范围的冻结使用**。没有重新绑定或伪造一份“新校准”。未传 `--dcn-config` 时保持原行为。

本次 engine SHA-256：`dc4200cd2bd4c21a443e05211d79599d3bf0fa5e2d55f4ae1ddef79c7a8ad08d`；DCN 库 SHA-256：`d22e5fa2ae9641ac2e8f0b847ce5aaea76f5fe445053ce99b2d448e21cd30982`。引擎与两份插件库必须配套记录；不同 GPU/TensorRT 环境应重新构建并验证。
