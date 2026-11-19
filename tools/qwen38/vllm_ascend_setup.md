# 在 CANN 9.0 镜像上跑 vLLM-Ascend（Qwen3.8-27B W8A8）

官方要求是 **CANN 9.1.0 + torch 2.10.0 + torch-npu 2.10.0.post4 + vllm-ascend 0.23.0**。
本机镜像是 CANN 9.0.0，本文记录把它推进到的**最远位置**以及每一步的实测结果。

**结论先说：服务能起、W8A8 权重能加载、KV 能分配、`/v1/models` 正常返回；但推理仍然
需要 `vllm_ascend_C` 扩展里的 GDN 自定义算子，而该扩展由 CANN 9.1 编译，在本镜像上
加载不了。** 用 torch 兜底补算子是一条可行的探索路径，但缺口不止一个。

## 实测进度

| 环节 | 结果 |
|---|---|
| `vllm 0.23.0` | 装 aarch64 wheel（`cp38-abi3`，cp311 可用） |
| `torch 2.10.0` + `torch_npu 2.10.0.post4` | `/workspace/venv-vllm`，NPU 识别为 Ascend910B2 |
| `vllm_ascend` 插件 | 正常激活，`NPUPlatform` |
| **W8A8 权重加载** | **成功，10 分片 28.01 GB**（`quantization=ascend`，msmodelslim 格式） |
| **KV cache** | **25.17 GiB**（`--max-model-len 8192 --gpu-memory-utilization 0.92`） |
| **HTTP 服务** | `Application startup complete`，冷启动约 **136 s** |
| `/v1/models` | 正常返回 |
| 首个推理请求 | **失败**，缺 GDN 自定义算子（见下） |

## 四个环境缺口与对应 workaround

1. **`libatb.so` 不存在**（CANN 9.0 toolkit 不含 ATB）
   → 装 `Ascend-cann-nnal_9.0.0_linux-aarch64.run --install --quiet
   --install-path=/workspace/nnal --whitelist=atb`。
   **必须先 `export ASCEND_TOOLKIT_HOME=<cann>`**，否则报 `Check owner failed`。

2. **算子命名空间不同**：PyPI 的 torch_npu 注册在 `torch.ops.npu`，
   vllm-ascend 硬编码 `torch.ops._C_ascend`。
   → `sitecustomize.py` 里做属性代理，把前者桥到后者。

3. **triton 缺 `next_power_of_2`**（PyPI 版没有该顶层工具函数）
   → 同一 shim 里补一个整数实现。

4. **依赖把 torch 升级到 2.11** → torch_npu 报
   `undefined symbol ... c10::MessageLogger`。
   → 回钉 `torch==2.10.0 / numpy==1.26.4 / triton==3.5.0`。
   另外 `torch.compile` 会在捕获阶段导入 `vllm_ascend_C`，故必须
   **`--enforce-eager`**。

## GDN 缺口：从运行期 dump 出来的真实契约

`vllm_ascend_C` 加载失败后，GDN 路径会依次撞到缺失算子。用桩函数把实参形状打印出来
后，已经补了两个（`/workspace/shim/sitecustomize.py`）：

**`npu_causal_conv1d_custom`**（实测实参）

```
out, x      (53, 10240) bf16        总 token 数 × conv 通道
weight      (4, 10240) bf16         kernel=4，权重按 [K, D] 传入
conv_state  (265, 3, 10240) bf16    分页缓冲：[block, K-1, D]
query_start_loc (2,) int32          分段边界 [0, 53]
cache_indices   (1, 1) int32        段 → block 的映射
activation_mode 1 (SiLU), pad_slot_id -1, run_mode 0
```

语义是「带分页初始状态的分段因果卷积」：`out[t] = Σ_k w[k]·cat(state, x)[t+k]`，
再乘 SiLU，最后把每段最后 `K-1` 个输入写回 `conv_state[block]`。

**`npu_fused_gdn_gating(A_log, a, b, dt_bias)`**

```
g = -exp(A_log) * softplus(a + dt_bias);  beta = sigmoid(b)
```

返回形状要是 `[B, T, H]`（flat token 输入时需要 `.unsqueeze(0)`，否则 vLLM 断言
`beta must be of shape [B, T, H]`）。

补完这两个之后，请求不再报 `AttributeError`，但**仍有后续缺口**（形状/语义层），
尚未跑到可用输出。

## 启动命令（可复现）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /workspace/nnal/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH="/workspace/shim:${PYTHONPATH}"     # 追加，别覆盖，否则丢 acl

/workspace/venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/Qwen3.8-27B-w8a8 \
  --served-model-name qwen3.8-27b-w8a8 \
  --max-model-len 8192 --gpu-memory-utilization 0.92 --enforce-eager --port 8001
```

## 判断

在 CANN 9.0 上把 vLLM-Ascend 补到可用，等于要**逐个重写 `vllm_ascend_C` 里的算子**
（卷积、gating，后面大概率还有 chunk/recurrent 的 GDN kernel），每个都要对齐分页状态和
变长 prefill 语义。这已经不是配置问题，而是重新实现一个后端。

**要拿到可用吞吐，正确做法是换官方 vllm-ascend 镜像**（CANN 9.1 + 匹配 torch_npu +
预编译扩展），本文件里的 4 个 workaround 在那种镜像上都不再需要。

同时保留另一条已验证的路线：**transformers + torch_npu eager**（见
`ascendc_gdn_decode/README.md`），它在本镜像上可以完整出结果，且已经用自定义
Ascend C 算子把 decode 提速 **1.123×**。
