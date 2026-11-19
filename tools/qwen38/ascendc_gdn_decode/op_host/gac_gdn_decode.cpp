#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/gac_gdn_decode_tiling.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr int64_t HK = 16;
constexpr int64_t HV = 48;
constexpr int64_t DK = 128;
constexpr int64_t DV = 128;
constexpr int64_t GROUP = HV / HK;

bool IsDenseFormat(ge::Format format) {
    // GE FormatAgnostic may label these dense shapes ND or NCHW; neither adds
    // padding. Never accept a transformed physical layout such as FRACTAL_NZ.
    return format == ge::FORMAT_ND || format == ge::FORMAT_NCHW;
}

bool ValidateShapes(const gert::Shape &q, const gert::Shape &k, const gert::Shape &v,
                    const gert::Shape &g, const gert::Shape &beta,
                    const gert::Shape &state, float l2Eps) {
    if (!std::isfinite(l2Eps) || l2Eps <= 0.0f) {
        return false;
    }
    if (q.GetDimNum() != 2 || q.GetDim(0) != HK || q.GetDim(1) != DK) {
        return false;
    }
    if (k.GetDimNum() != 2 || k.GetDim(0) != HK || k.GetDim(1) != DK) {
        return false;
    }
    if (v.GetDimNum() != 2 || v.GetDim(0) != HV || v.GetDim(1) != DV) {
        return false;
    }
    for (const gert::Shape *one : {&g, &beta}) {
        if (one->GetDimNum() != 1 || one->GetDim(0) != HV) {
            return false;
        }
    }
    if (state.GetDimNum() != 3 || state.GetDim(0) != HV ||
        state.GetDim(1) != DK || state.GetDim(2) != DV) {
        return false;
    }
    return true;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto *l2Eps = attrs->GetAttrPointer<float>(0);
    if (l2Eps == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // q, k, v are BF16; g, beta and state are FP32.
    const ge::DataType expected[6] = {
        ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT,
        ge::DT_FLOAT};
    for (size_t index = 0; index < 6; ++index) {
        if (context->GetInputShape(index) == nullptr ||
            context->GetInputDesc(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
        if (context->GetInputDesc(index)->GetDataType() != expected[index] ||
            !IsDenseFormat(context->GetInputDesc(index)->GetStorageFormat())) {
            return ge::GRAPH_FAILED;
        }
    }
    const ge::DataType outTypes[2] = {ge::DT_BF16, ge::DT_FLOAT};
    for (size_t index = 0; index < 2; ++index) {
        const auto *desc = context->GetOutputDesc(index);
        if (desc == nullptr || desc->GetDataType() != outTypes[index] ||
            !IsDenseFormat(desc->GetStorageFormat())) {
            return ge::GRAPH_FAILED;
        }
    }
    if (!ValidateShapes(context->GetInputShape(0)->GetStorageShape(),
                        context->GetInputShape(1)->GetStorageShape(),
                        context->GetInputShape(2)->GetStorageShape(),
                        context->GetInputShape(3)->GetStorageShape(),
                        context->GetInputShape(4)->GetStorageShape(),
                        context->GetInputShape(5)->GetStorageShape(),
                        *l2Eps)) {
        return ge::GRAPH_FAILED;
    }
    auto *tiling = context->GetTilingData<GacGdnDecodeTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->headCount = static_cast<uint32_t>(HV);
    tiling->headGroup = static_cast<uint32_t>(GROUP);
    tiling->l2Eps = *l2Eps;

    // One block per value head: deterministic coverage regardless of how many
    // vector cores the scheduler chooses to use.
    context->SetBlockDim(static_cast<uint32_t>(HV));
    context->GetRawTilingData()->SetDataSize(sizeof(GacGdnDecodeTilingData));
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return GRAPH_FAILED;
    }
    const auto *l2Eps = attrs->GetAttrPointer<float>(0);
    if (l2Eps == nullptr) {
        return GRAPH_FAILED;
    }
    for (size_t index = 0; index < 6; ++index) {
        if (context->GetInputShape(index) == nullptr) {
            return GRAPH_FAILED;
        }
    }
    if (!ValidateShapes(*context->GetInputShape(0), *context->GetInputShape(1),
                        *context->GetInputShape(2), *context->GetInputShape(3),
                        *context->GetInputShape(4), *context->GetInputShape(5),
                        *l2Eps)) {
        return GRAPH_FAILED;
    }
    auto *output = context->GetOutputShape(0);
    auto *stateOut = context->GetOutputShape(1);
    if (output == nullptr || stateOut == nullptr) {
        return GRAPH_FAILED;
    }
    output->SetDimNum(2);
    output->SetDim(0, HV);
    output->SetDim(1, DV);
    stateOut->SetDimNum(3);
    stateOut->SetDim(0, HV);
    stateOut->SetDim(1, DK);
    stateOut->SetDim(2, DV);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    if (context->GetInputDataType(0) != DT_BF16 ||
        context->GetInputDataType(1) != DT_BF16 ||
        context->GetInputDataType(2) != DT_BF16 ||
        context->GetInputDataType(3) != DT_FLOAT ||
        context->GetInputDataType(4) != DT_FLOAT ||
        context->GetInputDataType(5) != DT_FLOAT) {
        return GRAPH_FAILED;
    }
    if (context->SetOutputDataType(0, DT_BF16) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    return context->SetOutputDataType(1, DT_FLOAT);
}
}  // namespace ge

namespace ops {
class GacGdnDecode : public OpDef {
public:
    explicit GacGdnDecode(const char *name) : OpDef(name) {
        this->Input("q").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("k").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("v").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("g").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("beta").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("state").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("o").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("state_out").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("l2_eps").AttrType(REQUIRED).Float();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(GacGdnDecode);
}  // namespace ops
