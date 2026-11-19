// GacGdnDecode: fused decode step of the Qwen3.8-27B / Qwen3.5 Gated DeltaNet.
//
// Replaces the pure-PyTorch fallback `torch_recurrent_gated_delta_rule` for
// batch 1 and sequence length 1, where that path costs well over a thousand
// small tensor operations per layer per token. Everything stays in UB:
// L2 normalization of q/k (with the same BF16 rounding boundaries as PyTorch),
// the gated delta recurrence, and the state write-back.
//
// Contract (all shapes fixed, validated on host):
//   q, k       BF16 [HK=16, DK=128]
//   v          BF16 [HV=48, DV=128]
//   a, b       BF16 [HV]
//   a_log      FP32 [HV]
//   dt_bias    FP32 [HV]
//   state_in   FP32 [HV, DK, DV]
//   o          BF16 [HV, DV]
//   state_out  FP32 [HV, DK, DV]
//
// Per value head h (key head kh = h / 3):
//   decay  = exp(-exp(a_log[h]) * softplus(a[h] + dt_bias[h]))
//   beta   = sigmoid(b[h])
//   S      = state[h] * decay
//   kv     = S^T k_norm
//   delta  = (v[h] - kv) * beta
//   S      = S + k_norm (x) delta
//   o[h]   = S^T q_norm        with q_norm scaled by 1/sqrt(DK)
//
// No Cube use: the two reductions are register-resident accumulation over the
// 128 rows of one head's state tile.
#include "kernel_operator.h"
#include "gac_gdn_decode_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t HK = 16;
constexpr uint32_t HV = 48;
constexpr uint32_t DK = 128;
constexpr uint32_t DV = 128;
constexpr uint32_t GROUP = HV / HK;
constexpr uint32_t STATE_ELEMS = DK * DV;
// 1 / sqrt(128) as the reference computes it in FP32.
constexpr float Q_SCALE = 0.08838834764831845f;

class GacGdnDecodeKernel {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
                                GM_ADDR state, GM_ADDR o, GM_ADDR stateOut,
                                const GacGdnDecodeTilingData &tiling) {
        heads_ = tiling.headCount;
        group_ = tiling.headGroup;
        l2Eps_ = tiling.l2Eps;

        qGm_.SetGlobalBuffer((__gm__ bfloat16_t *)q, HK * DK);
        kGm_.SetGlobalBuffer((__gm__ bfloat16_t *)k, HK * DK);
        vGm_.SetGlobalBuffer((__gm__ bfloat16_t *)v, HV * DV);
        gGm_.SetGlobalBuffer((__gm__ float *)g, HV);
        betaGm_.SetGlobalBuffer((__gm__ float *)beta, HV);
        stateGm_.SetGlobalBuffer((__gm__ float *)state, HV * STATE_ELEMS);
        oGm_.SetGlobalBuffer((__gm__ bfloat16_t *)o, HV * DV);
        stateOutGm_.SetGlobalBuffer((__gm__ float *)stateOut, HV * STATE_ELEMS);

        pipe_.InitBuffer(stateBuf_, STATE_ELEMS * sizeof(float));
        pipe_.InitBuffer(qBuf_, DK * sizeof(float));
        pipe_.InitBuffer(kBuf_, DK * sizeof(float));
        pipe_.InitBuffer(vBuf_, DV * sizeof(float));
        pipe_.InitBuffer(tmpBuf_, DV * sizeof(float));
        pipe_.InitBuffer(accBuf_, DV * sizeof(float));
        pipe_.InitBuffer(oBuf_, DV * sizeof(float));
        pipe_.InitBuffer(sqBuf_, DV * sizeof(float));
        pipe_.InitBuffer(partBuf_, 8 * sizeof(float));
        pipe_.InitBuffer(scalBuf_, 8 * sizeof(float));
        pipe_.InitBuffer(onesBuf_, HV * sizeof(float));
        pipe_.InitBuffer(qBfBuf_, DK * sizeof(bfloat16_t));
        pipe_.InitBuffer(kBfBuf_, DK * sizeof(bfloat16_t));
        pipe_.InitBuffer(vBfBuf_, DV * sizeof(bfloat16_t));
        pipe_.InitBuffer(oBfBuf_, DV * sizeof(bfloat16_t));
        pipe_.InitBuffer(normBfBuf_, DK * sizeof(bfloat16_t));
        pipe_.InitBuffer(gBuf_, HV * sizeof(float));
        pipe_.InitBuffer(decayBuf_, HV * sizeof(float));
        pipe_.InitBuffer(betaBuf_, HV * sizeof(float));
    }

    __aicore__ inline void Process() {
        Duplicate(onesBuf_.Get<float>(), 1.0f, HV);
        PipeBarrier<PIPE_V>();
        PrepareGates();

        auto ready = pipe_.FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(ready);
        WaitFlag<HardEvent::V_S>(ready);

        auto decay = decayBuf_.Get<float>();
        auto beta = betaBuf_.Get<float>();
        // Exactly one value head per block. Making the mapping independent of
        // GetBlockNum() keeps every head covered even when the scheduler
        // launches fewer blocks than the tiling requests.
        const uint32_t h = GetBlockIdx();
        if (h < heads_) {
            ComputeHead(h, decay.GetValue(h), beta.GetValue(h));
        }
    }

private:
    // g (FP32) and beta (FP32) arrive precomputed from the model: the
    // reference forms g = -exp(a_log)*softplus(a + dt_bias) and
    // beta = bf16(b.sigmoid()) before the recurrence, so they are part of the
    // caller's contract, not of this operator.
    __aicore__ inline void PrepareGates() {
        DataCopy(gBuf_.Get<float>(), gGm_, HV);
        DataCopy(betaBuf_.Get<float>(), betaGm_, HV);
        auto loaded = pipe_.FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(loaded);
        WaitFlag<HardEvent::MTE2_V>(loaded);
        auto g = gBuf_.Get<float>();
        auto decay = decayBuf_.Get<float>();
        Exp(decay, g, HV);
        PipeBarrier<PIPE_V>();
    }

    // BF16-precision inverse L2 norm, matching the FLA/PyTorch reference:
    // each square and the partial sum are rounded to BF16, eps is applied at
    // BF16 precision, and the resulting inverse norm is BF16-rounded too.
    __aicore__ inline float InverseL2Norm(const LocalTensor<float> &x) {
        auto sq = sqBuf_.Get<float>();
        auto part = partBuf_.Get<float>();
        auto scal = scalBuf_.Get<float>();
        auto rounded = normBfBuf_.Get<bfloat16_t>();
        auto ones = onesBuf_.Get<float>();

        Mul(sq, x, x, DK);
        PipeBarrier<PIPE_V>();
        Cast(rounded, sq, RoundMode::CAST_RINT, DK);
        PipeBarrier<PIPE_V>();
        Cast(sq, rounded, RoundMode::CAST_NONE, DK);
        PipeBarrier<PIPE_V>();
        WholeReduceSum<float>(part, sq, 32, 4, 1, 1, 4);
        PipeBarrier<PIPE_V>();
        WholeReduceSum<float>(scal, part, 4, 1, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Cast(rounded, scal, RoundMode::CAST_RINT, 8);
        PipeBarrier<PIPE_V>();
        Cast(scal, rounded, RoundMode::CAST_NONE, 8);
        PipeBarrier<PIPE_V>();
        Adds(scal, scal, l2Eps_, 8);
        PipeBarrier<PIPE_V>();
        Cast(rounded, scal, RoundMode::CAST_RINT, 8);
        PipeBarrier<PIPE_V>();
        Cast(scal, rounded, RoundMode::CAST_NONE, 8);
        PipeBarrier<PIPE_V>();
        Sqrt(scal, scal, 8);
        PipeBarrier<PIPE_V>();
        Div(scal, ones, scal, 8);
        PipeBarrier<PIPE_V>();
        Cast(rounded, scal, RoundMode::CAST_RINT, 8);
        PipeBarrier<PIPE_V>();
        Cast(scal, rounded, RoundMode::CAST_NONE, 8);
        PipeBarrier<PIPE_V>();

        // Reading a UB scalar crosses V -> S and needs an explicit handshake.
        auto done = pipe_.FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(done);
        WaitFlag<HardEvent::V_S>(done);
        return scal.GetValue(0);
    }

    __aicore__ inline void ComputeHead(uint32_t h, float decay, float beta) {
        const uint32_t kh = h / group_;
        auto qF = qBuf_.Get<float>();
        auto kF = kBuf_.Get<float>();
        auto vF = vBuf_.Get<float>();
        auto state = stateBuf_.Get<float>();
        auto tmp = tmpBuf_.Get<float>();
        auto acc = accBuf_.Get<float>();
        auto out = oBuf_.Get<float>();
        auto qBf = qBfBuf_.Get<bfloat16_t>();
        auto kBf = kBfBuf_.Get<bfloat16_t>();
        auto vBf = vBfBuf_.Get<bfloat16_t>();
        auto oBf = oBfBuf_.Get<bfloat16_t>();

        // ---- load this head's q/k rows, v row and state tile.
        DataCopy(qBf, qGm_[kh * DK], DK);
        DataCopy(kBf, kGm_[kh * DK], DK);
        DataCopy(vBf, vGm_[h * DV], DV);
        DataCopy(state, stateGm_[h * STATE_ELEMS], STATE_ELEMS);
        auto mte2 = pipe_.FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(mte2);
        WaitFlag<HardEvent::MTE2_V>(mte2);
        Cast(qF, qBf, RoundMode::CAST_NONE, DK);
        Cast(kF, kBf, RoundMode::CAST_NONE, DK);
        Cast(vF, vBf, RoundMode::CAST_NONE, DV);
        PipeBarrier<PIPE_V>();

        // ---- q/k normalization. The reference normalizes in BF16, then scales
        // q in FP32, so the BF16 round-trip below is a real boundary.
        const float qInv = InverseL2Norm(qF);
        const float kInv = InverseL2Norm(kF);
        Muls(qF, qF, qInv, DK);
        Muls(kF, kF, kInv, DK);
        PipeBarrier<PIPE_V>();
        Cast(qBf, qF, RoundMode::CAST_RINT, DK);
        PipeBarrier<PIPE_V>();
        Cast(qF, qBf, RoundMode::CAST_NONE, DK);
        PipeBarrier<PIPE_V>();
        Cast(kBf, kF, RoundMode::CAST_RINT, DK);
        PipeBarrier<PIPE_V>();
        Cast(kF, kBf, RoundMode::CAST_NONE, DK);
        PipeBarrier<PIPE_V>();
        // The reference casts the normalized query to FP32 and then scales it;
        // no second BF16 rounding happens on q.
        Muls(qF, qF, Q_SCALE, DK);
        PipeBarrier<PIPE_V>();

        // ---- recurrence: decay, delta correction, then the output projection.
        Muls(state, state, decay, STATE_ELEMS);
        Duplicate(acc, 0.0f, DV);
        PipeBarrier<PIPE_V>();

        auto qkReady = pipe_.FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(qkReady);
        WaitFlag<HardEvent::V_S>(qkReady);

        for (uint32_t row = 0; row < DK; ++row) {
            Muls(tmp, state[row * DV], kF.GetValue(row), DV);
            Add(acc, acc, tmp, DV);
        }
        PipeBarrier<PIPE_V>();

        Sub(acc, vF, acc, DV);                    // v - S^T k
        PipeBarrier<PIPE_V>();
        Muls(acc, acc, beta, DV);                 // delta
        PipeBarrier<PIPE_V>();

        for (uint32_t row = 0; row < DK; ++row) {
            Muls(tmp, acc, kF.GetValue(row), DV);
            Add(state[row * DV], state[row * DV], tmp, DV);
        }
        PipeBarrier<PIPE_V>();

        Duplicate(out, 0.0f, DV);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < DK; ++row) {
            Muls(tmp, state[row * DV], qF.GetValue(row), DV);
            Add(out, out, tmp, DV);
        }
        PipeBarrier<PIPE_V>();

        auto store = pipe_.FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(store);
        WaitFlag<HardEvent::V_MTE3>(store);
        DataCopy(stateOutGm_[h * STATE_ELEMS], state, STATE_ELEMS);
        Cast(oBf, out, RoundMode::CAST_RINT, DV);
        PipeBarrier<PIPE_V>();
        DataCopy(oGm_[h * DV], oBf, DV);
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> stateBuf_, qBuf_, kBuf_, vBuf_, tmpBuf_, accBuf_, oBuf_;
    TBuf<QuePosition::VECCALC> sqBuf_, partBuf_, scalBuf_, onesBuf_;
    TBuf<QuePosition::VECCALC> qBfBuf_, kBfBuf_, vBfBuf_, oBfBuf_, normBfBuf_;
    TBuf<QuePosition::VECCALC> gBuf_, decayBuf_, betaBuf_;
    GlobalTensor<bfloat16_t> qGm_, kGm_, vGm_, oGm_;
    GlobalTensor<float> gGm_, betaGm_, stateGm_, stateOutGm_;
    uint32_t heads_, group_;
    float l2Eps_;
};
}  // namespace

extern "C" __global__ __aicore__ void gac_gdn_decode(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g, GM_ADDR beta,
    GM_ADDR state, GM_ADDR o, GM_ADDR stateOut, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GacGdnDecodeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    GacGdnDecodeKernel op;
    op.Init(q, k, v, g, beta, state, o, stateOut, tilingData);
    op.Process();
}
