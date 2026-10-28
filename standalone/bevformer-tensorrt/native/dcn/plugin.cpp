// SPDX-License-Identifier: Apache-2.0
// Narrow, opt-in DCNv2 plugin: validated FP16 im2col followed by default-mode HGEMM.
#include "NvInferRuntime.h"
#include "NvInferVersion.h"
#include "dcn_im2col.h"
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>

#if NV_TENSORRT_MAJOR != 10 || NV_TENSORRT_MINOR != 13
#error "GACModulatedDeformConv2d requires TensorRT 10.13 headers."
#endif

namespace gac::dcn {
namespace {
using namespace nvinfer1;
constexpr char kName[]{"GACModulatedDeformConv2d"};
constexpr char kVersion[]{"1"};
constexpr int kInputs = 4, kFields = 7;
constexpr char const* kNames[kFields]{"stride", "padding", "dilation", "group", "deformable_group",
                                      "block_threads", "specialize"};
constexpr int kLengths[kFields]{2, 2, 2, 1, 1, 1, 1};

int fail(char const* text) noexcept {
    std::fprintf(stderr, "[%s v%s] %s\n", kName, kVersion, text);
    return 1;
}

struct Options {
    // Match NVIDIA plugin field order [W,H]. Only symmetric pairs are validated.
    int32_t stride[2]{1, 1}, padding[2]{0, 0}, dilation[2]{1, 1};
    int32_t group{1}, deformableGroup{1}, blockThreads{128}, specialize{1};
    std::array<int32_t*, kFields> fields() noexcept {
        return {stride, padding, dilation, &group, &deformableGroup, &blockThreads, &specialize};
    }
};

bool validOptions(Options const& o) noexcept {
    return o.group == 1 && o.deformableGroup == 1
        && o.stride[0] > 0 && o.stride[0] == o.stride[1]
        && o.padding[0] >= 0 && o.padding[0] == o.padding[1]
        && o.dilation[0] > 0 && o.dilation[0] == o.dilation[1]
        && (o.blockThreads == 128 || o.blockThreads == 256 || o.blockThreads == 512 || o.blockThreads == 1024)
        && (o.specialize == 1 || o.specialize == 3 || o.specialize == 4);
}

bool sameDims(Dims const& a, Dims const& b) noexcept {
    if (a.nbDims != b.nbDims || a.nbDims <= 0 || a.nbDims > Dims::MAX_DIMS) return false;
    for (int i = 0; i < a.nbDims; ++i) if (a.d[i] != b.d[i]) return false;
    return true;
}

bool tensorFits(Dims const& d) noexcept {
    if (d.nbDims != 4) return false;
    int64_t count = 1;
    for (int i = 0; i < 4; ++i) {
        if (d.d[i] <= 0 || d.d[i] > std::numeric_limits<int32_t>::max() / count) return false;
        count *= d.d[i];
    }
    return static_cast<uint64_t>(count) <= static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(__half);
}

struct Shape { int c, h, w, co, ho, wo, spatial, k; size_t workspace; };

bool geometry(Dims const* in, Dims const* out, Options const& o, Shape& s) noexcept {
    if (!in || !validOptions(o)) return false;
    for (int i = 0; i < kInputs; ++i) if (!tensorFits(in[i])) return false;
    auto const& x = in[0]; auto const& offset = in[1];
    auto const& mask = in[2]; auto const& weight = in[3];
    if (x.d[0] != 1 || weight.d[1] != x.d[1] || weight.d[2] != 3 || weight.d[3] != 3) return false;
    int64_t const hNumerator = x.d[2] + 2LL * o.padding[1] - 2LL * o.dilation[1] - 1;
    int64_t const wNumerator = x.d[3] + 2LL * o.padding[0] - 2LL * o.dilation[0] - 1;
    if (hNumerator < 0 || wNumerator < 0) return false;
    int64_t const ho = hNumerator / o.stride[1] + 1;
    int64_t const wo = wNumerator / o.stride[0] + 1;
    constexpr int64_t limit = std::numeric_limits<int32_t>::max();
    if (ho > limit || wo > limit / ho) return false;
    // Match the native kernel's int32 coordinate-intermediate bounds, not only
    // its final tensor sizes. Huge dilation/stride must fail before enqueue.
    for (int axis = 0; axis < 2; ++axis) {
        int64_t const output = axis == 0 ? wo : ho;
        int64_t const extent = 2LL * o.dilation[axis];
        int64_t const lastOrigin = (output - 1) * o.stride[axis];
        if (extent > limit || lastOrigin > limit || lastOrigin - o.padding[axis] + extent > limit) return false;
    }
    int64_t const spatial = ho * wo;
    if (x.d[1] > limit / 9 || 9 * x.d[1] > limit / spatial) return false;
    if (offset.d[0] != 1 || offset.d[1] != 18 || offset.d[2] != ho || offset.d[3] != wo
        || mask.d[0] != 1 || mask.d[1] != 9 || mask.d[2] != ho || mask.d[3] != wo) return false;
    Dims expected{}; expected.nbDims = 4;
    expected.d[0] = 1; expected.d[1] = weight.d[0]; expected.d[2] = ho; expected.d[3] = wo;
    if (!tensorFits(expected) || (out && !sameDims(expected, *out))) return false;
    int64_t const elements = 9 * x.d[1] * spatial;
    if (static_cast<uint64_t>(elements) > static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max()) / sizeof(__half)) return false;
    s = {static_cast<int>(x.d[1]), static_cast<int>(x.d[2]), static_cast<int>(x.d[3]),
         static_cast<int>(weight.d[0]), static_cast<int>(ho), static_cast<int>(wo),
         static_cast<int>(spatial), static_cast<int>(9 * x.d[1]),
         static_cast<size_t>(elements) * sizeof(__half)};
    return true;
}

bool descriptors(PluginTensorDesc const* in, int ni, PluginTensorDesc const* out, int no,
                 Options const& o, Shape& s) noexcept {
    if (!in || !out || ni != kInputs || no != 1) return false;
    std::array<Dims, kInputs> dims;
    for (int i = 0; i < kInputs; ++i) {
        if (in[i].type != DataType::kHALF || in[i].format != TensorFormat::kLINEAR) return false;
        dims[i] = in[i].dims;
    }
    return out[0].type == DataType::kHALF && out[0].format == TensorFormat::kLINEAR
        && geometry(dims.data(), &out[0].dims, o, s);
}

// One resource per TRT execution context, shared by its DCN layer clones.
// Weak references never keep an engine/context alive. Distinct TRT contexts
// receive distinct handles; no process-global mutable cuBLAS handle is used.
struct ContextBlas {
    CUcontext owner{};
    cublasHandle_t handle{};
    std::mutex mutex;
    explicit ContextBlas(CUcontext context) : owner(context) {
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasCreate failed");
        if (cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS
            || cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST) != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(handle); handle = nullptr;
            throw std::runtime_error("Could not configure default-math host-pointer cuBLAS handle");
        }
    }
    ~ContextBlas() noexcept {
        if (!handle) return;
        CUcontext current{};
        bool const queried = cuCtxGetCurrent(&current) == CUDA_SUCCESS;
        bool const push = !queried || current != owner;
        if (push && cuCtxPushCurrent(owner) != CUDA_SUCCESS) {
            fail("Owner CUDA context unavailable during cuBLAS resource release");
            return;
        }
        if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) fail("cublasDestroy failed");
        if (push) { CUcontext previous{}; if (cuCtxPopCurrent(&previous) != CUDA_SUCCESS) fail("cuCtxPopCurrent failed"); }
    }
};

std::shared_ptr<ContextBlas> acquireBlas(IPluginResourceContext* resource) {
    if (!resource) throw std::runtime_error("attachToContext requires a TensorRT resource context");
    CUcontext context{};
    if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context)
        throw std::runtime_error("No current CUDA context during plugin attachment");
    using Key = std::pair<uintptr_t, uintptr_t>;
    static std::mutex registryMutex;
    static std::map<Key, std::weak_ptr<ContextBlas>> registry;
    std::lock_guard<std::mutex> guard(registryMutex);
    for (auto it = registry.begin(); it != registry.end(); ) {
        if (it->second.expired()) it = registry.erase(it); else ++it;
    }
    Key const key{reinterpret_cast<uintptr_t>(resource), reinterpret_cast<uintptr_t>(context)};
    auto existing = registry[key].lock();
    if (existing) return existing;
    auto created = std::make_shared<ContextBlas>(context);
    registry[key] = created;
    return created;
}

class DCNPlugin final : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuild,
                        public IPluginV3OneRuntime {
public:
    explicit DCNPlugin(Options options) noexcept : mOptions(options) {
        auto values = mOptions.fields();
        for (int i = 0; i < kFields; ++i)
            mFields[i] = PluginField{kNames[i], values[i], PluginFieldType::kINT32, kLengths[i]};
        mCollection = PluginFieldCollection{kFields, mFields.data()};
    }
    DCNPlugin(DCNPlugin const&) = delete;
    DCNPlugin& operator=(DCNPlugin const&) = delete;
    ~DCNPlugin() override = default;
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        default: return nullptr;
        }
    }
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    IPluginV3* clone() noexcept override {
        // Build/deserialized clones have no execution-context resource yet.
        auto* result = new (std::nothrow) DCNPlugin(mOptions);
        if (!result) fail("Plugin clone allocation failed");
        return result;
    }
    IPluginV3* attachToContext(IPluginResourceContext* resource) noexcept override {
        try {
            auto result = std::make_unique<DCNPlugin>(mOptions);
            result->mBlas = acquireBlas(resource);
            return result.release();
        } catch (std::exception const& error) { fail(error.what()); return nullptr; }
        catch (...) { fail("Unknown resource attachment failure"); return nullptr; }
    }
    PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &mCollection; }
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* io, int32_t ni, int32_t no) noexcept override {
        return io && ni == kInputs && no == 1 && pos >= 0 && pos < kInputs + 1
            && io[pos].desc.type == DataType::kHALF && io[pos].desc.format == TensorFormat::kLINEAR;
    }
    int32_t getOutputDataTypes(DataType* out, int32_t no, DataType const* in, int32_t ni) const noexcept override {
        if (!in || !out || ni != kInputs || no != 1) return fail("Expected four execution inputs and one output");
        // Weakly typed ONNX initially describes FP32 tensors. The full-model
        // precision policy constrains plugin boundaries after parsing. Preserve
        // that logical type here; executable format negotiation below is HALF only.
        if (in[0] != DataType::kHALF && in[0] != DataType::kFLOAT) return fail("Expected floating logical inputs");
        for (int i = 1; i < kInputs; ++i) if (in[i] != in[0]) return fail("Logical DCN input types must agree");
        out[0] = in[0];
        return 0;
    }
    int32_t getOutputShapes(DimsExprs const* in, int32_t ni, DimsExprs const*, int32_t ns,
                           DimsExprs* out, int32_t no, IExprBuilder& builder) noexcept override {
        if (!in || !out || ni != kInputs || no != 1 || ns != 0) return fail("Invalid DCN shape ABI");
        std::array<Dims, kInputs> dims;
        for (int i = 0; i < kInputs; ++i) {
            if (in[i].nbDims != 4) return fail("Only static NCHW rank-four inputs are supported");
            dims[i].nbDims = 4;
            for (int j = 0; j < 4; ++j) {
                if (!in[i].d[j] || !in[i].d[j]->isConstant()) return fail("Dynamic DCN shapes are unsupported");
                dims[i].d[j] = in[i].d[j]->getConstantValue();
            }
        }
        Shape s{};
        if (!geometry(dims.data(), nullptr, mOptions, s)) return fail("Unsupported DCN shape/options");
        out[0].nbDims = 4;
        out[0].d[0] = builder.constant(1); out[0].d[1] = builder.constant(s.co);
        out[0].d[2] = builder.constant(s.ho); out[0].d[3] = builder.constant(s.wo);
        for (int i = 0; i < 4; ++i) if (!out[0].d[i]) return fail("Output shape expression allocation failed");
        return 0;
    }
    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t ni,
                            DynamicPluginTensorDesc const* out, int32_t no) noexcept override {
        if (!in || !out || ni != kInputs || no != 1) return fail("Invalid DCN descriptors");
        std::array<PluginTensorDesc, kInputs> input;
        for (int i = 0; i < kInputs + 1; ++i) {
            auto const& d = i < kInputs ? in[i] : out[0];
            if (!sameDims(d.desc.dims, d.min) || !sameDims(d.desc.dims, d.opt) || !sameDims(d.desc.dims, d.max))
                return fail("DCN MIN/OPT/MAX must equal the static descriptor");
            if (i < kInputs) input[i] = d.desc;
        }
        Shape s{};
        return descriptors(input.data(), ni, &out[0].desc, no, mOptions, s) ? 0 : fail("Unsupported DCN descriptors");
    }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* in, int32_t ni,
                            DynamicPluginTensorDesc const* out, int32_t no) const noexcept override {
        if (!in || !out || ni != kInputs || no != 1) return 0;
        std::array<PluginTensorDesc, kInputs> input;
        for (int i = 0; i < kInputs; ++i) input[i] = in[i].desc;
        Shape s{};
        return descriptors(input.data(), ni, &out[0].desc, no, mOptions, s) ? s.workspace : 0;
    }
    int32_t onShapeChange(PluginTensorDesc const* in, int32_t ni, PluginTensorDesc const* out, int32_t no) noexcept override {
        Shape s{};
        return descriptors(in, ni, out, no, mOptions, s) ? 0 : fail("Unsupported DCN runtime descriptors");
    }
    int32_t enqueue(PluginTensorDesc const* in, PluginTensorDesc const* out, void const* const* inputs,
                    void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
        Shape s{};
        if (!descriptors(in, kInputs, out, 1, mOptions, s) || !inputs || !outputs || !outputs[0] || !workspace || !mBlas)
            return fail("Invalid DCN enqueue arguments or missing attached cuBLAS resource");
        for (int i = 0; i < kInputs; ++i) if (!inputs[i]) return fail("Null DCN input");
        CUcontext current{};
        if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != mBlas->owner)
            return fail("DCN enqueued under a different CUDA context than attachment");
        int const launched = dcn_im2col(inputs[0], inputs[1], inputs[2], workspace, 1, s.c, s.h, s.w,
            s.ho, s.wo, 3, 3, mOptions.padding[1], mOptions.padding[0], mOptions.stride[1], mOptions.stride[0],
            mOptions.dilation[1], mOptions.dilation[0], 1, GAC_DCN_FLOAT16,
            mOptions.blockThreads, mOptions.specialize, stream);
        if (launched) {
            std::fprintf(stderr, "[%s] DCN im2col launch status=%d\n", kName, launched);
            return launched;
        }
        try {
            // Lock covers stream mutation and host submission only. No GPU wait.
            std::lock_guard<std::mutex> guard(mBlas->mutex);
            auto status = cublasSetStream(mBlas->handle, stream);
            if (status != CUBLAS_STATUS_SUCCESS) return fail("cublasSetStream failed");
            __half const alpha = __float2half(1.0F), beta = __float2half(0.0F);
            // Row-major W[Co,K] * col[K,Ho*Wo] is column-major col^T * W^T.
            status = cublasHgemm(mBlas->handle, CUBLAS_OP_N, CUBLAS_OP_N, s.spatial, s.co, s.k,
                &alpha, static_cast<__half const*>(workspace), s.spatial,
                static_cast<__half const*>(inputs[3]), s.k, &beta, static_cast<__half*>(outputs[0]), s.spatial);
            if (status != CUBLAS_STATUS_SUCCESS) {
                std::fprintf(stderr, "[%s] cublasHgemm status=%d\n", kName, static_cast<int>(status));
                return 1;
            }
            return 0;
        } catch (std::exception const& error) { return fail(error.what()); }
        catch (...) { return fail("Unknown DCN cuBLAS submission failure"); }
    }
private:
    Options mOptions;
    std::array<PluginField, kFields> mFields{};
    PluginFieldCollection mCollection{};
    std::shared_ptr<ContextBlas> mBlas;
};

class DCNCreator final : public IPluginCreatorV3One {
public:
    DCNCreator() noexcept {
        for (int i = 0; i < kFields; ++i) mFields[i] = PluginField{kNames[i], nullptr, PluginFieldType::kINT32, kLengths[i]};
        mCollection = PluginFieldCollection{kFields, mFields.data()};
    }
    char const* getPluginName() const noexcept override { return kName; }
    char const* getPluginVersion() const noexcept override { return kVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mCollection; }
    IPluginV3* createPlugin(char const*, PluginFieldCollection const* fields, TensorRTPhase phase) noexcept override {
        if (!fields || fields->nbFields < 5 || fields->nbFields > kFields || !fields->fields
            || (phase != TensorRTPhase::kBUILD && phase != TensorRTPhase::kRUNTIME)) {
            fail("Invalid DCN field collection"); return nullptr;
        }
        Options options;
        auto values = options.fields();
        std::array<bool, kFields> seen{};
        for (int i = 0; i < fields->nbFields; ++i) {
            auto const& field = fields->fields[i];
            int index = -1;
            if (field.name) for (int j = 0; j < kFields; ++j) if (!std::strcmp(field.name, kNames[j])) { index = j; break; }
            if (index < 0 || seen[index] || !field.data || field.type != PluginFieldType::kINT32 || field.length != kLengths[index]) {
                fail("Unknown/duplicate DCN field or invalid INT32 field length"); return nullptr;
            }
            std::memcpy(values[index], field.data, sizeof(int32_t) * kLengths[index]);
            seen[index] = true;
        }
        for (int i = 0; i < 5; ++i) if (!seen[i]) { fail("Missing DCN geometry field"); return nullptr; }
        if (!validOptions(options) || (phase == TensorRTPhase::kRUNTIME && fields->nbFields != kFields)) {
            fail("Unsupported DCN options or incomplete serialized fields"); return nullptr;
        }
        auto* result = new (std::nothrow) DCNPlugin(options);
        if (!result) fail("DCN plugin allocation failed");
        return result;
    }
private:
    std::array<PluginField, kFields> mFields{};
    PluginFieldCollection mCollection{};
};

REGISTER_TENSORRT_PLUGIN(DCNCreator);
} // namespace
} // namespace gac::dcn
