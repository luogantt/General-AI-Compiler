# Qwen3.8-27B 在昇腾上的 vLLM-Ascend 部署与实测

本文记录两条路线：**① CANN 9.2.1 官方镜像 + vLLM-Ascend 0.30.0（已跑通，含实测数据）**，
以及 **② CANN 9.0 镜像上的失败尝试（保留作对照）**。

---

## 一、跑通的配置（CANN 9.2.1）

| 项 | 值 |
|---|---|
| 硬件 | 单卡 Ascend 910B（64 GB HBM），同机还有第二张卡未启用 |
| CANN | **9.2.1**（`/usr/local/Ascend/cann-9.2.1`） |
| 驱动 | 25.5.0 |
| 镜像自带 | `torch 2.10.0+cpu`、`torch_npu 2.10.0.post7`、`vllm 0.30.0`（editable `/vllm-workspace/vllm`）、`vllm_ascend`（`/vllm-workspace/vllm-ascend`）、`transformers 5.14.1`、`triton 3.6.0` |
| 模型 | `Eco-Tech/Qwen3.8-27B-w8a8`（msmodelslim W8A8，30.39 GB / 25 文件，逐文件 SHA-256 校验） |
| 权重加载 | **28.02 GB** |
| KV cache | **25.01 GiB** |
| 启动耗时 | 约 **180 s**（含权重加载与 KV profile） |

**这个镜像里组件都是预装的，不需要 pip 安装**。我们早期用 venv 从 PyPI 装的那套反而
失败：PyPI 直连超时，清华源对 aarch64 wheel 返回 403。最终直接用系统解释器
`/usr/local/python3.12.16/bin/python3.12` 就跑通了。

### 两个必须的启动条件

1. **ATB 必须在 `LD_LIBRARY_PATH` 里**，否则 `torch_npu` 加载 `libop_plugin_atb.so` 时报
   `libatb.so: cannot open shared object file`：
   ```bash
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```
   （本机实际路径 `/usr/local/Ascend/nnal/atb/9.2.1/atb/cxx_abi_1/lib/libatb.so`）

2. **GDN 类模型的 mamba 状态 dtype 必须是 bfloat16**：
   ```bash
   --mamba-ssm-cache-dtype bfloat16
   ```
   默认 FP32 initial state 只在 Ascend950 支持；第三方报告在 SGLang 上复现过同一问题。

### 启动命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
export ASCEND_RT_VISIBLE_DEVICES=0

/usr/local/python3.12.16/bin/python3.12 -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/Qwen3.8-27B-w8a8 \
  --served-model-name qwen3.8-27b-w8a8 \
  --max-model-len 32768 --gpu-memory-utilization 0.92 \
  --mamba-ssm-cache-dtype bfloat16 \
  --port 8001
```

`/v1/chat/completions` 与 `/v1/completions` 均正常，中文输出正确。

---

## 二、实测性能（单卡 910B，W8A8；prompt ≈200 token，输出 128 token）

| 并发 | 输出吞吐 | TTFT p50 | TPOT p50 | 单序列速度 |
|---:|---:|---:|---:|---:|
| **1** | **34.11 tok/s** | 117.8 ms | 28.54 ms | 35.03 tok/s |
| 4 | 121.04 tok/s | 294.2 ms | 30.93 ms | 32.33 tok/s |
| 8 | **218.18 tok/s** | 466.4 ms | 33.20 ms | 30.12 tok/s |

长上下文（≈4K token 输入）：

| 场景 | TTFT p50 | 输出吞吐 |
|---|---:|---:|
| 4K 输入，并发 1，输出 64 | 2108.7 ms | 16.34 tok/s |
| 4K 输入，并发 4，输出 64 | 369.2 ms（命中前缀缓存） | 108.3 tok/s |

运行后显存占用 **59535 / 65536 MB（90.8%）**，与 `--gpu-memory-utilization 0.92` 吻合。

### 与其它路线的对照

| 路线 | 单卡输出速度 | 说明 |
|---|---:|---|
| transformers + torch_npu **eager**，BF16 | 5.04 → 5.66 tok/s | batch=1，无服务化；5.66 是加上本仓库 `GacGdnDecode` 算子后 |
| **vLLM-Ascend 0.30.0 + W8A8** | **34.11 tok/s（并发 1）** | 本次实测 |
| vLLM-Ascend，并发 8 | 218.18 tok/s | 本次实测 |
| 第三方 SGLang，910B3 **×4 TP4** | 127.5 ~ 545 tok/s | 4 卡 + 128K 输入 + 90% 前缀命中，口径不同，不可直接比 |

**结论**：单卡上 vLLM 相对 transformers eager 提升约 **6 倍**（5.66 → 34.11 tok/s）。
收益来自图执行和算子替换，而不是量化本身。

---

## 三、失败记录：CANN 9.0 镜像（保留作对照）

在另一台 **CANN 9.0.0** 的机器上（单卡 910B2），用 venv 从 PyPI 装
`vllm 0.23.0 + vllm-ascend 0.23.0`，推进到"服务能起、权重能加载、KV 25.17 GiB、
`/v1/models` 正常"，但**首个推理请求必然失败**：缺 `vllm_ascend_C` 扩展提供的 GDN 算子。

| 缺口 | 现象 | 处理 |
|---|---|---|
| `libatb.so` | `torch_npu` 加载失败 | 装 `Ascend-cann-nnal 9.0.0 --whitelist=atb`（需先 `export ASCEND_TOOLKIT_HOME`） |
| 算子命名空间 | PyPI 的 torch_npu 注册在 `torch.ops.npu`，vllm-ascend 访问 `torch.ops._C_ascend` | `sitecustomize.py` 属性代理桥接 |
| `triton.next_power_of_2` | PyPI triton 无该工具函数 | shim 补整数实现 |
| **`vllm_ascend_C`** | `undefined symbol: rtFunctionGetMetaInfoSize`（CANN 9.1 编译） | **无法绕过**，需重写 `npu_causal_conv1d_custom` 等算子 |
| 依赖升级 | vllm 把 torch 升到 2.11，torch_npu 报 `undefined symbol ... c10::MessageLogger` | 回钉 torch 2.10.0 / numpy 1.26.4 / triton 3.5.0 |

用桩函数 dump 出的 GDN 缺失算子真实契约（对自研 Ascend C 算子有用）：

```text
npu_causal_conv1d_custom(out, x, weight, conv_state=..., bias_opt=...,
    query_start_loc_opt=..., cache_indices_opt=..., activation_mode=1, pad_slot_id=-1, run_mode=0)
  out, x          (tokens, 10240) bf16
  weight          (4, 10240) bf16            kernel = 4
  conv_state      (blocks, 3, 10240) bf16    分页状态
  query_start_loc (2,) int32                 分段边界
npu_fused_gdn_gating(A_log, a, b, dt_bias) -> (g, beta)    形状需 [B, T, H]
```

**教训**：在 CANN 9.0 上补全 vLLM 等于自己重写一个后端；正确做法是用官方镜像。
