# GacGdnDecode 鈥?Ascend C 瀹炵幇鐨?Gated DeltaNet decode 绠楀瓙

Qwen3.8-27B 鐨?64 灞傞噷鏈?**48 灞傛槸 Gated DeltaNet**銆倀ransformers 鍦ㄦ病鏈?`flash-linear-attention` / `causal-conv1d` 鏃惰蛋 `torch_recurrent_gated_delta_rule`
鐨勭函 PyTorch 鍥為€€锛氭瘡灞傛瘡 token 涓婂崈娆″皬寮犻噺鎿嶄綔锛屾槸鍗曞崱 910B2 涓?decode 鐨勪富瑕?杞欢寮€閿€銆傝繖涓畻瀛愭妸鍏朵腑**涓€瓒熷畬鏁寸殑 decode 姝?*鏀跺埌涓€涓?Ascend C kernel 閲屻€?
## 瀹冪畻浠€涔?
鍥哄畾濂戠害锛坆atch 1銆佸簭鍒楅暱搴?1銆丵wen3.8-27B 灏哄锛夛紝鍏ㄩ儴鍦?UB 鍐呭畬鎴愶細

```
q, k    BF16 [16,128]        16 涓?key head
v       BF16 [48,128]        48 涓?value head
g       FP32 [48]            妯″瀷宸茬畻濂界殑 log 琛板噺
beta    FP32 [48]            妯″瀷宸茬畻濂界殑 bf16 sigmoid 缁撴灉
state   FP32 [48,128,128]    閫掓帹鐘舵€?鈫?o         BF16 [48,128]
  state_out FP32 [48,128,128]
```

瀵规瘡涓?value head `h`锛坘ey head `kh = h / 3`锛夛細

```
decay  = exp(g[h])
beta   = beta[h]
S      = state[h] * decay
kv     = S岬€ 路 l2norm(k[kh])
delta  = (v[h] 鈭?kv) 路 beta
S      = S + l2norm(k[kh]) 鈯?delta
o[h]   = S岬€ 路 (l2norm(q[kh]) / sqrt(128))
```

涓や釜鍏抽敭鐐癸細

1. **鑸嶅叆杈圭晫瀵归綈 PyTorch**銆傚弬鑰冨疄鐜扮殑 L2 norm 鏄湪 BF16 寮犻噺涓婄畻鐨勶紝鎵€浠ュ钩鏂广€?   閮ㄥ垎鍜屻€乣+eps`銆乣rsqrt`銆佸綊涓€鍖栫粨鏋滈兘瑕佸悇鑷粡杩?BF16 鑸嶅叆锛沗beta` 鏉ヨ嚜
   `b.sigmoid()` 鐨?BF16 缁撴灉锛屼篃宸茬粡甯︾潃涓€娆¤垗鍏ャ€俴ernel 閲岀敤
   `Cast(..., CAST_RINT)` 鏄惧紡澶嶇幇杩欎簺杈圭晫銆?2. **涓嶇敤 Cube**銆備袱娆″綊绾﹂兘鏄湪瀵勫瓨鍣ㄥ唴鎸?128 琛岀疮鍔狅紝涓棿閲忕暀鍦?UB銆?
## 鐩綍

| 鏂囦欢 | 浣滅敤 |
|---|---|
| `GacGdnDecode.json` | msopgen 鐨勭畻瀛愭弿杩?|
| `op_host/gac_gdn_decode.cpp` | OpDef銆両nferShape/InferDataType銆乀iling |
| `op_kernel/gac_gdn_decode.cpp` | Ascend C kernel |
| `op_kernel/gac_gdn_decode_tiling.h` | host/device 鍏变韩鐨?POD |
| `torch_bridge.cpp` | TorchNPU `OpCommand::RunOpApi` 妗ユ帴锛岃蛋 aclnn |
| `build_opp.py` | 鐢?msopgen + CMake 鐢熸垚骞舵墦鍖?OPP锛堜笉瑁呰繘 CANN锛?|
| `build_bridge.py` | 缂栬瘧骞跺姞杞?`.so`锛岃褰?ABI 涓庡搱甯?|
| `test_gdn_decode.py` | 鏁板€?+ 寤惰繜娴嬭瘯 |

## 鏋勫缓涓庨獙璇?
```bash
export GDN=/workspace/gdn_op
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 1) OPP锛堣緭鍑虹洰褰曞繀椤讳笉瀛樺湪锛?python build_opp.py --output $GDN/build/opp --soc Ascend910B2 --jobs 8
RUN=$(ls $GDN/build/opp/build_out/*.run)
bash "$RUN" --install-path=$GDN/install

# 2) torch bridge
V=$GDN/install/vendors/gacgdn
export ASCEND_CUSTOM_OPP_PATH=$GDN/install:$V
export ASCEND_CACHE_PATH=$GDN/cache          # 鎹㈠€欓€夊氨鎹㈡柊缂撳瓨鐩綍
python build_bridge.py --opapi-include-dir $V/op_api/include \
  --opapi-lib-dir $V/op_api/lib --custom-opp-path $GDN/install \
  --build-dir $GDN/bridge --jobs 8 --load

# 3) 鏁板€?+ 寤惰繜
python test_gdn_decode.py --bridge $GDN/bridge/lib/gac_qwen38_gdn_decode_bridge.so
```

`ASCEND_CUSTOM_OPP_PATH` 蹇呴』鍦?*瀵煎叆 torch_npu 涔嬪墠**璁剧疆锛屽苟涓旀瘡娆℃崲 OPP 鍊欓€?閮借鏂板缓 `ASCEND_CACHE_PATH` 鍜岃繘绋嬶紝鍚﹀垯浼氬懡涓棫鏍哥殑缂栬瘧缂撳瓨銆?
## 瀹炴祴锛堝崟鍗?Ascend 910B2 64GB锛孋ANN 9.0.0锛宼orch 2.7.1锛?
鏁板€硷紙3 缁勯殢鏈鸿緭鍏ワ紝鍚?scale 0.25/1/4锛夛細

| 杈撳嚭 | max_abs | rel_l2 |
|---|---|---|
| `o`锛圔F16锛?| 1.95e-03 | 5.96e-03 |
| `state`锛團P32锛?| 1.10e-02 | 3.68e-04 ~ 1.08e-03 |

`o` 鐨勬渶澶ц宸害 0.25 涓?BF16 ULP锛屽凡缁忔槸杈撳嚭绮惧害鐨勯噺绾э紱`state` 鏈?55鈥?3% 鐨?鍏冪礌閫愪綅鐩稿悓銆傛畫宸潵鑷?FP32 褰掔害椤哄簭涓?BF16 杈圭晫涓婄殑涓€娆?ulp 宸紓锛屼笉鏄€昏緫閿欒銆?
20 姝ヨ繛缁€掓帹锛堟妸 `state_out` 鍥炵亴锛?*涓嶅彂鏁?*锛歚state` 鐩稿璇樊绋冲畾鍦?2鈥?e-3锛宍o` 鐨?max_abs 濮嬬粓 鈮?7.3e-4銆?
鍗曟璋冪敤锛?8 涓?head锛夛細**0.04鈥?.11 ms**锛屽嵆涓€灞?GDN 鐨?decode 姝ワ紱48 灞傜害
2鈥? ms/token銆?
绔埌绔紙鎶婄畻瀛愭帴鍥?`Qwen3_5GatedDeltaNet.recurrent_gated_delta_rule`锛?64 token 鐢熸垚锛孉/B/B/A 浜ゆ浛涓よ疆锛夛細

| 閰嶇疆 | tok/s锛? 娆★級 | 涓綅鏁?|
|---|---|---|
| 鍘熷 PyTorch 鍥為€€ | 5.18 / 5.01 / 5.02 / 5.06 | 5.04 |
| GacGdnDecode | 5.59 / 5.64 / 5.87 / 5.68 | **5.66** |

**1.123脳**锛屼笖涓ゆ patched 杩愯鐨?48 涓柊 token 瀹屽叏涓€鑷达紝涓?baseline 涔熶竴鑷淬€?鏀剁泭鏈夐檺鐨勭洿鎺ュ師鍥狅細GDN 鍙槸 64 灞備腑鐨?48 灞傜殑**涓€閮ㄥ垎**锛屽悓灞傜殑
`in_proj_qkv / conv1d / out_proj` 绛夊ぇ绠楀瓙浠嶆槸涓诲鎴愭湰銆?
## 璋冭瘯杩囩▼涓慨鎺夌殑涓や釜鐪熼棶棰?
1. **`ones` 缂撳啿瓒婄晫**锛歚beta = 1/(1+exp(-b))` 鐢?`Div(beta, ones, beta, 48)`
   闇€瑕?48 涓?1锛屼絾缂撳啿鍙?`Duplicate` 浜?8 涓紝绗?8 涓箣鍚庤鍒拌秺鐣屾暟鎹紙0锛夛紝
   瀵艰嚧 `beta[h鈮?] = 0`锛屾洿鏂伴」鎭掍负闆躲€傜幇璞℃槸"鍓?8 涓?head 姝ｇ‘銆佸叾浣欏叏 0"銆?2. **block 鏄犲皠**锛氭渶鍒濈敤 `for (h = GetBlockIdx(); h < heads_; h += GetBlockNum())`
   璺ㄦ鍒嗛厤锛屼絾鏁寸綉鍙湁 8 涓?block 鐪熸琚皟搴︼紝`GetBlockNum()` 鍙堣繑鍥炰簡鏇村ぇ鐨勫€硷紝
   浜庢槸鍙鐩栦簡 head 0鈥?銆傛敼鎴?`SetBlockDim(48)` + 涓€涓?block 涓€涓?head 鍚庣‘瀹氬彲澶嶇幇銆?
## 杈圭晫

- 鍙敮鎸?**decode**锛圱=1銆乥atch 1锛夈€俻refill 璧?`chunk_gated_delta_rule`锛屾湰绠楀瓙涓嶈鐩栥€?- 灏哄鍐欐涓?Qwen3.8-27B 鐨?16/48/128/128锛沨ost 渚т弗鏍兼牎楠岋紝褰㈢姸涓嶇鐩存帴澶辫触锛屼笉闈欓粯鍥為€€銆?- 鐘舵€佹寜 FP32 杩涘嚭锛涙ā鍨嬬紦瀛橀噷瀛樼殑鏄?BF16锛屾帴鍥炴ā鍨嬫椂鍦ㄨˉ涓佸眰鍋氫竴娆′笂杞€?- 涓€涓?block 涓€涓?head 鏄负浜嗙‘瀹氳鐩栵紱鐗╃悊 AIV 灏戜簬 48 鏃堕潬澶氭尝娆¤皟搴︼紝鏈仛璐熻浇鍧囪　銆?
## 与 vLLM 路线的关系

本算子走的是 **transformers + torch_npu eager** 路线：不需要 CANN 9.1，也不需要
`vllm_ascend_C` 扩展，decode 从 5.04 → 5.66 tok/s（1.123×），输出 token 逐位一致。

服务化路线另有实测：在 **CANN 9.2.1 官方镜像 + vLLM-Ascend 0.30.0 + W8A8** 上，
单卡 910B 达到 **34.11 tok/s（并发 1）/ 218.18 tok/s（并发 8）**，约为 eager 路线的
**6 倍**。完整命令、参数和失败记录见 [../vllm_ascend_setup.md](../vllm_ascend_setup.md)。

两条路线的关系：vLLM 是生产形态的吞吐基线，本算子解决的是"没有 vLLM 时如何把
GDN 的碎片算子收进一个 Ascend C kernel"，两者互为对照。