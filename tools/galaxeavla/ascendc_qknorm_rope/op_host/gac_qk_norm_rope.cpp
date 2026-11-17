#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/gac_qk_norm_rope_tiling.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr int64_t SEQUENCE = 32;
constexpr int64_t WIDTH = 256;
constexpr int64_t ROTARY = 64;

bool ValidateShapes(const gert::Shape &x, const gert::Shape &weight,
                    const gert::Shape &cos, const gert::Shape &sin,
                    int64_t headStride, float epsilon) {
    if ((headStride != 512 && headStride != 256) ||
        !std::isfinite(epsilon) || epsilon <= 0.0f) {
        return false;
    }
    const int64_t projectionWidth = headStride == 512 ? 4096 : 512;
    if (x.GetDimNum() != 3 || x.GetDim(0) != 1 ||
        x.GetDim(1) != SEQUENCE || x.GetDim(2) != projectionWidth ||
        weight.GetDimNum() != 1 || weight.GetDim(0) != WIDTH) {
        return false;
    }
    if (cos.GetDimNum() != 4 || sin.GetDimNum() != 4) {
        return false;
    }
    for (size_t index = 0; index < 4; ++index) {
        if (cos.GetDim(index) != sin.GetDim(index)) {
            return false;
        }
    }
    return cos.GetDim(0) == 1 && cos.GetDim(1) == 1 &&
           cos.GetDim(2) >= SEQUENCE && cos.GetDim(3) == ROTARY;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto *headStride = attrs->GetAttrPointer<int64_t>(0);
    const auto *epsilon = attrs->GetAttrPointer<float>(1);
    if (headStride == nullptr || epsilon == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < 4; ++index) {
        if (context->GetInputShape(index) == nullptr || context->GetInputDesc(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const auto expectedType = index == 0 ? ge::DT_BF16 : ge::DT_FLOAT;
        // GE FormatAgnostic may label these unchanged dense shapes NCHW.
        // Neither accepted format adds blocked storage/padding. Do not accept
        // FRACTAL_NZ, NC1HWC0, or other transformed physical layouts here.
        const auto storageFormat = context->GetInputDesc(index)->GetStorageFormat();
        if (context->GetInputDesc(index)->GetDataType() != expectedType ||
            (storageFormat != ge::FORMAT_ND && storageFormat != ge::FORMAT_NCHW)) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto *outputDesc = context->GetOutputDesc(0);
    if (outputDesc == nullptr || outputDesc->GetDataType() != ge::DT_BF16 ||
        (outputDesc->GetStorageFormat() != ge::FORMAT_ND &&
         outputDesc->GetStorageFormat() != ge::FORMAT_NCHW)) {
        return ge::GRAPH_FAILED;
    }
    if (!ValidateShapes(context->GetInputShape(0)->GetStorageShape(),
                        context->GetInputShape(1)->GetStorageShape(),
                        context->GetInputShape(2)->GetStorageShape(),
                        context->GetInputShape(3)->GetStorageShape(),
                        *headStride, *epsilon)) {
        return ge::GRAPH_FAILED;
    }
    auto *tiling = context->GetTilingData<GacQkNormRopeTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->headCount = *headStride == 512 ? 8 : 2;
    tiling->headStride = static_cast<uint32_t>(*headStride);
    tiling->projectionWidth = tiling->headCount * tiling->headStride;
    tiling->rowCount = tiling->headCount * SEQUENCE;
    tiling->epsilon = *epsilon;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t vectorCores = platform.GetCoreNumAiv();
    if (vectorCores == 0) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(std::min(vectorCores, tiling->rowCount));
    context->GetRawTilingData()->SetDataSize(sizeof(GacQkNormRopeTilingData));
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
    const auto *headStride = attrs->GetAttrPointer<int64_t>(0);
    const auto *epsilon = attrs->GetAttrPointer<float>(1);
    if (headStride == nullptr || epsilon == nullptr) {
        return GRAPH_FAILED;
    }
    for (size_t index = 0; index < 4; ++index) {
        if (context->GetInputShape(index) == nullptr) {
            return GRAPH_FAILED;
        }
    }
    if (!ValidateShapes(*context->GetInputShape(0), *context->GetInputShape(1),
                        *context->GetInputShape(2), *context->GetInputShape(3),
                        *headStride, *epsilon)) {
        return GRAPH_FAILED;
    }
    auto *output = context->GetOutputShape(0);
    if (output == nullptr) {
        return GRAPH_FAILED;
    }
    output->SetDimNum(4);
    output->SetDim(0, 1);
    output->SetDim(1, *headStride == 512 ? 8 : 2);
    output->SetDim(2, SEQUENCE);
    output->SetDim(3, WIDTH);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    if (context->GetInputDataType(0) != DT_BF16 ||
        context->GetInputDataType(1) != DT_FLOAT ||
        context->GetInputDataType(2) != DT_FLOAT ||
        context->GetInputDataType(3) != DT_FLOAT) {
        return GRAPH_FAILED;
    }
    return context->SetOutputDataType(0, DT_BF16);
}
}  // namespace ge

namespace ops {
class GacQkNormRope : public OpDef {
public:
    explicit GacQkNormRope(const char *name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("cos").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("sin").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("head_stride").AttrType(REQUIRED).Int();
        this->Attr("epsilon").AttrType(REQUIRED).Float();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(GacQkNormRope);
}  // namespace ops
