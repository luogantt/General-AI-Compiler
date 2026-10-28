// SPDX-License-Identifier: Apache-2.0
// GAC's independent implementation of the TensorRT 10.13 IPluginV3 contract.
// API reference: NVIDIA/TensorRT release/10.13/include/NvInferRuntime.h.
#include "NvInferRuntime.h"
#include "NvInferVersion.h"
#include "msda_kernel.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>

#if NV_TENSORRT_MAJOR != 10 || NV_TENSORRT_MINOR != 13
#error "GACMSDeformAttn is built and validated against TensorRT 10.13 headers."
#endif

namespace gac::msda
{
namespace
{
using namespace nvinfer1;
constexpr char kName[]{"GACMSDeformAttn"};
constexpr char kVersion[]{"1"};
constexpr int32_t kInputs{5};
constexpr int32_t kOutputs{1};
constexpr int32_t kFields{4};
constexpr char const* kFieldNames[kFields]{"channels_per_thread", "block_threads", "specialize", "prefetch"};

struct Options
{
    int32_t channelsPerThread{1};
    int32_t blockThreads{256};
    int32_t specialize{1}; // 0: generic, 1: auto, 2: require L4/P8/C32, 3: guarded index32 auto.
    int32_t prefetch{0}; // 0: none, 1: geometry, 2: geometry + four-neighbor Value.
};

int32_t fail(char const* message) noexcept
{
    std::fprintf(stderr, "[%s v%s] %s\n", kName, kVersion, message);
    return 1;
}

bool validOptions(Options const& options) noexcept
{
    return (options.channelsPerThread == 1 || options.channelsPerThread == 2 || options.channelsPerThread == 4)
        && (options.blockThreads == 128 || options.blockThreads == 256 || options.blockThreads == 512
            || options.blockThreads == 768)
        && options.specialize >= 0 && options.specialize <= 3 && options.prefetch >= 0 && options.prefetch <= 2;
}

bool floatingType(DataType type) noexcept
{
    return type == DataType::kFLOAT || type == DataType::kHALF;
}

bool equalDims(Dims const& left, Dims const& right) noexcept
{
    if (left.nbDims != right.nbDims || left.nbDims <= 0 || left.nbDims > Dims::MAX_DIMS)
    {
        return false;
    }
    for (int32_t axis = 0; axis < left.nbDims; ++axis)
    {
        if (left.d[axis] != right.d[axis])
        {
            return false;
        }
    }
    return true;
}

bool concreteTensor(Dims const& dims, int32_t rank) noexcept
{
    if (dims.nbDims != rank)
    {
        return false;
    }
    // Bound byte offsets as well as each extent. All supported types use <= 4 bytes.
    int64_t count{1};
    constexpr auto kMaxElements = std::numeric_limits<int64_t>::max() / 4;
    for (int32_t axis = 0; axis < rank; ++axis)
    {
        auto const extent = dims.d[axis];
        if (extent <= 0 || extent > std::numeric_limits<int32_t>::max() || count > kMaxElements / extent)
        {
            return false;
        }
        count *= extent;
    }
    return true;
}

bool validShapes(Dims const* in, Dims const* out, Options const& options) noexcept
{
    constexpr int32_t ranks[kInputs]{4, 2, 1, 6, 5};
    for (int32_t index = 0; index < kInputs; ++index)
    {
        if (!concreteTensor(in[index], ranks[index]))
        {
            return false;
        }
    }
    auto const& value = in[0];
    auto const& shapes = in[1];
    auto const& starts = in[2];
    auto const& locations = in[3];
    auto const& weights = in[4];
    if (shapes.d[1] != 2 || locations.d[5] != 2 || shapes.d[0] != starts.d[0]
        || shapes.d[0] != locations.d[3] || value.d[0] != locations.d[0] || value.d[2] != locations.d[2])
    {
        return false;
    }
    for (int32_t axis = 0; axis < weights.nbDims; ++axis)
    {
        if (weights.d[axis] != locations.d[axis])
        {
            return false;
        }
    }
    if (options.specialize == 2 && (shapes.d[0] != 4 || locations.d[4] != 8 || value.d[3] != 32))
    {
        return false;
    }
    Dims expected{};
    expected.nbDims = 4;
    expected.d[0] = value.d[0];
    expected.d[1] = locations.d[1];
    expected.d[2] = value.d[2];
    expected.d[3] = value.d[3];
    if (!concreteTensor(expected, 4) || (out != nullptr && !equalDims(expected, *out)))
    {
        return false;
    }
    return true;
}

bool validDescriptors(PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out,
    int32_t nbOutputs, Options const& options) noexcept
{
    if (in == nullptr || out == nullptr || nbInputs != kInputs || nbOutputs != kOutputs || !validOptions(options))
    {
        return false;
    }
    std::array<Dims, kInputs> dims{};
    for (int32_t index = 0; index < kInputs; ++index)
    {
        dims[index] = in[index].dims;
        auto const expected = index == 1 || index == 2 ? DataType::kINT32 : in[0].type;
        if (in[index].type != expected || in[index].format != TensorFormat::kLINEAR)
        {
            return false;
        }
    }
    return floatingType(in[0].type) && out[0].type == in[0].type && out[0].format == TensorFormat::kLINEAR
        && validShapes(dims.data(), &out[0].dims, options);
}

class MSDAPlugin final : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild,
                         public IPluginV3OneRuntime
{
public:
    explicit MSDAPlugin(Options options) noexcept : mOptions{options}
    {
        // Each clone owns these values and pointers; never copy a PluginFieldCollection.
        int32_t const* values[kFields]{&mOptions.channelsPerThread, &mOptions.blockThreads, &mOptions.specialize,
            &mOptions.prefetch};
        for (int32_t index = 0; index < kFields; ++index)
        {
            mFields[index] = PluginField{kFieldNames[index], values[index], PluginFieldType::kINT32, 1};
        }
        mCollection = PluginFieldCollection{kFields, mFields.data()};
    }

    MSDAPlugin(MSDAPlugin const&) = delete;
    MSDAPlugin& operator=(MSDAPlugin const&) = delete;
    ~MSDAPlugin() override = default;

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override
    {
        switch (type)
        {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        default: return nullptr;
        }
    }

    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    int32_t getNbOutputs() const noexcept override { return kOutputs; }

    IPluginV3* clone() noexcept override
    {
        auto* result = new (std::nothrow) MSDAPlugin{mOptions};
        if (result == nullptr)
        {
            fail("clone allocation failed");
        }
        return result;
    }

    IPluginV3* attachToContext(IPluginResourceContext*) noexcept override { return clone(); }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mCollection; }

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (inOut == nullptr || nbInputs != kInputs || nbOutputs != kOutputs || pos < 0 || pos >= kInputs + kOutputs)
        {
            return false;
        }
        // TensorRT only guarantees that positions through pos are initialized.
        auto const& desc = inOut[pos].desc;
        if (desc.format != TensorFormat::kLINEAR)
        {
            return false;
        }
        if (pos == 1 || pos == 2)
        {
            return desc.type == DataType::kINT32;
        }
        return floatingType(desc.type) && (pos == 0 || desc.type == inOut[0].desc.type);
    }

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        if (outputTypes == nullptr || inputTypes == nullptr || nbInputs != kInputs || nbOutputs != kOutputs
            || !floatingType(inputTypes[0]) || inputTypes[1] != DataType::kINT32 || inputTypes[2] != DataType::kINT32
            || inputTypes[3] != inputTypes[0] || inputTypes[4] != inputTypes[0])
        {
            return fail("expected FLOAT/HALF value, INT32 shapes/starts, and matching locations/weights");
        }
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*, int32_t nbShapeInputs,
        DimsExprs* outputs, int32_t nbOutputs, IExprBuilder&) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != kInputs || nbOutputs != kOutputs || nbShapeInputs != 0)
        {
            return fail("expected five execution inputs, no shape inputs, and one output");
        }
        std::array<Dims, kInputs> dims{};
        for (int32_t index = 0; index < kInputs; ++index)
        {
            dims[index].nbDims = inputs[index].nbDims;
            if (dims[index].nbDims <= 0 || dims[index].nbDims > Dims::MAX_DIMS)
            {
                return fail("invalid input rank");
            }
            for (int32_t axis = 0; axis < dims[index].nbDims; ++axis)
            {
                auto const* expr = inputs[index].d[axis];
                if (expr == nullptr || !expr->isConstant())
                {
                    return fail("dynamic shapes are unsupported; all input dimensions must be static");
                }
                dims[index].d[axis] = expr->getConstantValue();
            }
        }
        if (!validShapes(dims.data(), nullptr, mOptions))
        {
            return fail("invalid MSDA input dimensions or incompatible required specialization");
        }
        outputs[0].nbDims = 4;
        outputs[0].d[0] = inputs[0].d[0];
        outputs[0].d[1] = inputs[3].d[1];
        outputs[0].d[2] = inputs[0].d[2];
        outputs[0].d[3] = inputs[0].d[3];
        return 0;
    }

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
        DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override
    {
        if (in == nullptr || out == nullptr || nbInputs != kInputs || nbOutputs != kOutputs)
        {
            return fail("invalid number of inputs or outputs");
        }
        std::array<PluginTensorDesc, kInputs> descriptors{};
        for (int32_t index = 0; index < kInputs + kOutputs; ++index)
        {
            auto const& tensor = index < kInputs ? in[index] : out[index - kInputs];
            if (!equalDims(tensor.desc.dims, tensor.min) || !equalDims(tensor.desc.dims, tensor.opt)
                || !equalDims(tensor.desc.dims, tensor.max))
            {
                return fail("dynamic profiles are unsupported; MIN/OPT/MAX must equal the static dimensions");
            }
            if (index < kInputs)
            {
                descriptors[index] = tensor.desc;
            }
        }
        return validDescriptors(descriptors.data(), nbInputs, &out[0].desc, nbOutputs, mOptions)
            ? 0 : fail("invalid static MSDA shapes, formats, data types, or required specialization");
    }

    size_t getWorkspaceSize(DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t)
        const noexcept override
    {
        return 0;
    }

    int32_t onShapeChange(PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override
    {
        return validDescriptors(in, nbInputs, out, nbOutputs, mOptions)
            ? 0 : fail("unsupported runtime MSDA descriptor; only consistent, nonempty static tensors are supported");
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        if (!validDescriptors(inputDesc, kInputs, outputDesc, kOutputs, mOptions)
            || inputs == nullptr || outputs == nullptr || outputs[0] == nullptr)
        {
            return fail("invalid enqueue descriptors or pointers");
        }
        for (int32_t index = 0; index < kInputs; ++index)
        {
            if (inputs[index] == nullptr)
            {
                return fail("null enqueue input pointer");
            }
        }
        auto const& value = inputDesc[0].dims;
        auto const& locations = inputDesc[3].dims;
        // shapes and starts are device tensors. Never read them on the host, copy
        // them to the host, or synchronize the caller's stream here.
        int32_t const status = gac_msda_forward(inputs[0], static_cast<int32_t const*>(inputs[1]),
            static_cast<int32_t const*>(inputs[2]), inputs[3], inputs[4], outputs[0], value.d[0], value.d[1],
            value.d[2], value.d[3], locations.d[3], locations.d[1], locations.d[4],
            inputDesc[0].type == DataType::kFLOAT ? GAC_MSDA_FLOAT32 : GAC_MSDA_FLOAT16,
            mOptions.channelsPerThread, mOptions.blockThreads,
            mOptions.specialize, mOptions.prefetch, stream);
        if (status != 0)
        {
            std::fprintf(stderr, "[%s v%s] CUDA launch failed, status=%d\n", kName, kVersion, status);
        }
        return status;
    }

private:
    Options mOptions;
    std::array<PluginField, kFields> mFields{};
    PluginFieldCollection mCollection{};
};

class MSDACreator final : public IPluginCreatorV3One
{
public:
    MSDACreator() noexcept
    {
        for (int32_t index = 0; index < kFields; ++index)
        {
            mFields[index] = PluginField{kFieldNames[index], nullptr, PluginFieldType::kINT32, 1};
        }
        mCollection = PluginFieldCollection{kFields, mFields.data()};
    }

    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mCollection; }

    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fields, TensorRTPhase phase) noexcept override
    {
        if (fields == nullptr || fields->nbFields < 0 || fields->nbFields > kFields
            || (fields->nbFields != 0 && fields->fields == nullptr)
            || (phase != TensorRTPhase::kBUILD && phase != TensorRTPhase::kRUNTIME))
        {
            fail("invalid plugin field collection or TensorRT phase");
            return nullptr;
        }
        Options options{};
        int32_t* values[kFields]{&options.channelsPerThread, &options.blockThreads, &options.specialize, &options.prefetch};
        std::array<bool, kFields> seen{};
        for (int32_t index = 0; index < fields->nbFields; ++index)
        {
            auto const& field = fields->fields[index];
            if (field.name == nullptr || field.data == nullptr || field.type != PluginFieldType::kINT32 || field.length != 1)
            {
                fail("every plugin field must contain exactly one INT32 value");
                return nullptr;
            }
            int32_t matched{-1};
            for (int32_t option = 0; option < kFields; ++option)
            {
                if (std::strcmp(field.name, kFieldNames[option]) == 0)
                {
                    matched = option;
                    break;
                }
            }
            if (matched < 0 || seen[matched])
            {
                fail("unknown or duplicate plugin field");
                return nullptr;
            }
            // PluginField storage does not promise int32 alignment.
            std::memcpy(values[matched], field.data, sizeof(int32_t));
            seen[matched] = true;
        }
        if ((phase == TensorRTPhase::kRUNTIME && fields->nbFields != kFields) || !validOptions(options))
        {
            fail("invalid tuning options or incomplete serialized runtime fields");
            return nullptr;
        }
        auto* plugin = new (std::nothrow) MSDAPlugin{options};
        if (plugin == nullptr)
        {
            fail("plugin allocation failed");
        }
        return plugin;
    }

private:
    std::array<PluginField, kFields> mFields{};
    PluginFieldCollection mCollection{};
};

// Loading this shared library registers only GAC's own plugin name.
REGISTER_TENSORRT_PLUGIN(MSDACreator);
} // namespace
} // namespace gac::msda
