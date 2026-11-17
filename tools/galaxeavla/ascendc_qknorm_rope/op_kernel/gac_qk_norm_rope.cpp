// Experimental LIBERO AE FP32 Q/K RMSNorm + partial RoPE.
// No Cube use: intermediates stay in UB; output is stored directly in BNSD.
// Build with FP contraction disabled. The two BF16 casts are intentional
// rounding boundaries, not redundant conversions to remove.
#include "kernel_operator.h"
#include "gac_qk_norm_rope_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t WIDTH = 256;
constexpr uint32_t SEQUENCE = 32;
constexpr uint32_t ROTARY = 64;
constexpr uint32_t HALF_ROTARY = 32;

class GacQkNormRopeKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR cos,
                              GM_ADDR sin, GM_ADDR y,
                              const GacQkNormRopeTilingData &tiling) {
        heads_ = tiling.headCount;
        headStride_ = tiling.headStride;
        projectionWidth_ = tiling.projectionWidth;
        rows_ = tiling.rowCount;
        epsilon_ = tiling.epsilon;
        xGm_.SetGlobalBuffer((__gm__ bfloat16_t *)x, SEQUENCE * projectionWidth_);
        weightGm_.SetGlobalBuffer((__gm__ float *)weight, WIDTH);
        // Only the first 32 positions are read, even when S_cos > 32.
        cosGm_.SetGlobalBuffer((__gm__ float *)cos, SEQUENCE * ROTARY);
        sinGm_.SetGlobalBuffer((__gm__ float *)sin, SEQUENCE * ROTARY);
        yGm_.SetGlobalBuffer((__gm__ bfloat16_t *)y, rows_ * WIDTH);

        pipe_.InitBuffer(xQueue_, 1, WIDTH * sizeof(bfloat16_t));
        pipe_.InitBuffer(weightQueue_, 1, WIDTH * sizeof(float));
        pipe_.InitBuffer(ropeQueue_, 1, 2 * ROTARY * sizeof(float));
        pipe_.InitBuffer(yQueue_, 1, WIDTH * sizeof(bfloat16_t));
        pipe_.InitBuffer(xFloatBuffer_, WIDTH * sizeof(float));
        pipe_.InitBuffer(squareBuffer_, WIDTH * sizeof(float));
        pipe_.InitBuffer(partialSumBuffer_, 8 * sizeof(float));
        pipe_.InitBuffer(scaleBuffer_, 8 * sizeof(float));
        pipe_.InitBuffer(onesBuffer_, 8 * sizeof(float));
        pipe_.InitBuffer(roundedNormBuffer_, WIDTH * sizeof(bfloat16_t));
        pipe_.InitBuffer(rotatedBuffer_, ROTARY * sizeof(float));
        pipe_.InitBuffer(ropeProductBuffer_, ROTARY * sizeof(float));
    }

    __aicore__ inline void Process() {
        auto ones = onesBuffer_.Get<float>();
        Duplicate(ones, 1.0f, 8);
        PipeBarrier<PIPE_V>();
        auto weight = weightQueue_.AllocTensor<float>();
        DataCopy(weight, weightGm_, WIDTH);
        weightQueue_.EnQue(weight);
        weight = weightQueue_.DeQue<float>();
        // Original Qwen3_5RMSNorm uses (1 + weight.float()), not weight.
        Adds(weight, weight, 1.0f, WIDTH);
        PipeBarrier<PIPE_V>();

        for (uint32_t row = GetBlockIdx(); row < rows_; row += GetBlockNum()) {
            const uint32_t head = row / SEQUENCE;
            const uint32_t position = row % SEQUENCE;
            CopyIn(position, head);
            Compute(weight, ones);
            CopyOut(row);
        }
        weightQueue_.FreeTensor(weight);
    }

private:
    __aicore__ inline void CopyIn(uint32_t position, uint32_t head) {
        auto x = xQueue_.AllocTensor<bfloat16_t>();
        // Q skips the gate half of each packed 512-wide head. K is dense.
        DataCopy(x, xGm_[position * projectionWidth_ + head * headStride_], WIDTH);
        xQueue_.EnQue(x);
        auto rope = ropeQueue_.AllocTensor<float>();
        DataCopy(rope, cosGm_[position * ROTARY], ROTARY);
        DataCopy(rope[ROTARY], sinGm_[position * ROTARY], ROTARY);
        ropeQueue_.EnQue(rope);
    }

    __aicore__ inline void Compute(const LocalTensor<float> &gamma,
                                  const LocalTensor<float> &ones) {
        auto x = xQueue_.DeQue<bfloat16_t>();
        auto rope = ropeQueue_.DeQue<float>();
        auto xf = xFloatBuffer_.Get<float>();
        auto squares = squareBuffer_.Get<float>();
        auto partial = partialSumBuffer_.Get<float>();
        auto scale = scaleBuffer_.Get<float>();
        auto rounded = roundedNormBuffer_.Get<bfloat16_t>();
        auto rotated = rotatedBuffer_.Get<float>();
        auto product = ropeProductBuffer_.Get<float>();

        Cast(xf, x, RoundMode::CAST_NONE, WIDTH);
        PipeBarrier<PIPE_V>();
        xQueue_.FreeTensor(x);
        Mul(squares, xf, xf, WIDTH);
        PipeBarrier<PIPE_V>();
        // Two FP32 binary-tree reductions: 4 groups of 64, then 4 totals.
        // This can differ in last bits from the PyTorch reduction tree.
        WholeReduceSum<float>(partial, squares, 64, 4, 1, 1, 8);
        Duplicate(scale, 0.0f, 8);
        PipeBarrier<PIPE_V>();
        WholeReduceSum<float>(scale, partial, 4, 1, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Muls(scale, scale, 0.00390625f, 8);  // Exact FP32 1/256; no device scalar cast.
        PipeBarrier<PIPE_V>();
        Adds(scale, scale, epsilon_, 8);
        PipeBarrier<PIPE_V>();
        // Ascend C documents that intrinsic Rsqrt can miss FP32 accuracy
        // targets on A2. Use the recommended Sqrt + Div high-precision path.
        Sqrt(scale, scale, 8);
        PipeBarrier<PIPE_V>();
        Div(scale, ones, scale, 8);
        PipeBarrier<PIPE_V>();
        const event_t scalarEvent = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(scalarEvent);
        WaitFlag<HardEvent::V_S>(scalarEvent);
        const float inverseRms = scale.GetValue(0);
        Muls(xf, xf, inverseRms, WIDTH);
        PipeBarrier<PIPE_V>();
        Mul(xf, xf, gamma, WIDTH);
        PipeBarrier<PIPE_V>();

        // Required BF16 round-to-nearest-even boundary after RMSNorm.
        Cast(rounded, xf, RoundMode::CAST_RINT, WIDTH);
        PipeBarrier<PIPE_V>();
        Cast(xf, rounded, RoundMode::CAST_NONE, WIDTH);
        PipeBarrier<PIPE_V>();

        // rotate_half(first 64) == [-x[32:64], x[0:32]].
        Muls(rotated, xf[HALF_ROTARY], -1.0f, HALF_ROTARY);
        Muls(rotated[HALF_ROTARY], xf, 1.0f, HALF_ROTARY);
        PipeBarrier<PIPE_V>();
        // Keep two separately rounded FP32 multiplies followed by addition.
        Mul(product, rotated, rope[ROTARY], ROTARY);
        Mul(xf, xf, rope, ROTARY);
        PipeBarrier<PIPE_V>();
        Add(xf, xf, product, ROTARY);
        PipeBarrier<PIPE_V>();
        auto y = yQueue_.AllocTensor<bfloat16_t>();
        // Tail 192 is the previously rounded normalization result unchanged.
        Cast(y, xf, RoundMode::CAST_RINT, WIDTH);
        PipeBarrier<PIPE_V>();
        yQueue_.EnQue(y);
        ropeQueue_.FreeTensor(rope);
    }

    __aicore__ inline void CopyOut(uint32_t row) {
        auto y = yQueue_.DeQue<bfloat16_t>();
        DataCopy(yGm_[row * WIDTH], y, WIDTH);
        yQueue_.FreeTensor(y);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> xQueue_, weightQueue_, ropeQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TBuf<QuePosition::VECCALC> xFloatBuffer_, squareBuffer_;
    TBuf<QuePosition::VECCALC> partialSumBuffer_, scaleBuffer_, onesBuffer_;
    TBuf<QuePosition::VECCALC> roundedNormBuffer_, rotatedBuffer_, ropeProductBuffer_;
    GlobalTensor<bfloat16_t> xGm_, yGm_;
    GlobalTensor<float> weightGm_, cosGm_, sinGm_;
    uint32_t heads_, headStride_, projectionWidth_, rows_;
    float epsilon_;
};
}  // namespace

extern "C" __global__ __aicore__ void gac_qk_norm_rope(
    GM_ADDR x, GM_ADDR weight, GM_ADDR cos, GM_ADDR sin, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GacQkNormRopeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    GacQkNormRopeKernel op;
    op.Init(x, weight, cos, sin, y, tilingData);
    op.Process();
}
