// SPDX-License-Identifier: Apache-2.0
// Numerical reference: NVIDIA TensorRT release/10.13, Apache-2.0:
// https://github.com/NVIDIA/TensorRT/blob/release/10.13/plugin/
// multiscaleDeformableAttnPlugin/multiscaleDeformableIm2ColCuda.cuh
// Original copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES.
// Original implementation derived from Deformable DETR (SenseTime,
// copyright 2020-2023) and DCN (Microsoft, copyright 2018-2023).
// This implementation shares sampling work across channels, with guarded
// offsets and an optional checked 32-bit index path. FP16 explicitly reproduces
// the fused half operations measured
// in NVIDIA TensorRT 10.13.3 on SM86, while preserving its pairwise interpolation
// tree and sequential point accumulation. Revalidate against other binaries.

#ifndef GAC_MSDA_BUILD
#define GAC_MSDA_BUILD
#endif
#include "msda_kernel.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

struct Dimensions {
    int64_t batch, spatial, heads, channels, levels, queries, points;
};

// _rn fixes the rounding boundaries that must remain separate. Explicit half
// FMA fixes the contractions in the measured NVIDIA binary: coordinates, the
// first product of each interpolation pair, and the attention accumulation.
// Merely using ordinary __hadd/__hmul lets nvcc choose different contractions.
template <typename T> struct Scalar;
template <> struct Scalar<float> {
    __device__ static float zero() { return 0.0F; }
    __device__ static float one() { return 1.0F; }
    __device__ static float nan() { return __int_as_float(0x7fc00000); }
    __device__ static float from_int(int32_t x) { return __int2float_rn(x); }
    __device__ static float add(float a, float b) { return __fadd_rn(a, b); }
    __device__ static float sub(float a, float b) { return __fsub_rn(a, b); }
    __device__ static float mul(float a, float b) { return __fmul_rn(a, b); }
    __device__ static float fma(float a, float b, float c) { return __fmaf_rn(a, b, c); }
    __device__ static float coordinate(float x, float extent) {
        return sub(mul(x, extent), 0.5F);
    }
    __device__ static bool inside(float x, float extent) {
        return x > -1.0F && x < extent;
    }
    __device__ static int32_t floor_int(float x) { return __float2int_rd(x); }
};
template <> struct Scalar<__half> {
    __device__ static __half zero() { return __ushort_as_half(0); }
    __device__ static __half one() { return __ushort_as_half(0x3c00); }
    __device__ static __half nan() { return __ushort_as_half(0x7e00); }
    __device__ static __half from_int(int32_t x) { return __int2half_rd(x); }
    __device__ static __half add(__half a, __half b) { return __hadd_rn(a, b); }
    __device__ static __half sub(__half a, __half b) { return __hsub_rn(a, b); }
    __device__ static __half mul(__half a, __half b) { return __hmul_rn(a, b); }
    __device__ static __half fma(__half a, __half b, __half c) { return __hfma(a, b, c); }
    __device__ static __half coordinate(__half x, __half extent) {
        return fma(x, extent, __ushort_as_half(0xb800));
    }
    __device__ static bool inside(__half x, __half extent) {
        return __hgt(x, __ushort_as_half(0xbc00)) && __hlt(x, extent);
    }
    __device__ static int32_t floor_int(__half x) { return __half2int_rd(x); }
};

template <typename T, int Width> struct Pack;
template <typename T> struct Pack<T, 1> {
    using Type = T;
    __device__ static Type broadcast(T x) { return x; }
    __device__ static Type load(const T* ptr) { return *ptr; }
    __device__ static void store(T* ptr, Type x) { *ptr = x; }
    __device__ static Type add(Type a, Type b) { return Scalar<T>::add(a, b); }
    __device__ static Type mul(Type a, Type b) { return Scalar<T>::mul(a, b); }
    __device__ static Type fma(Type a, Type b, Type c) { return Scalar<T>::fma(a, b, c); }
};
template <> struct Pack<__half, 2> {
    using Type = __half2;
    __device__ static Type broadcast(__half x) { return __half2half2(x); }
    __device__ static Type load(const __half* ptr) {
        return *reinterpret_cast<const __half2*>(ptr);
    }
    __device__ static void store(__half* ptr, Type x) {
        *reinterpret_cast<__half2*>(ptr) = x;
    }
    __device__ static Type add(Type a, Type b) { return __hadd2_rn(a, b); }
    __device__ static Type mul(Type a, Type b) { return __hmul2_rn(a, b); }
    __device__ static Type fma(Type a, Type b, Type c) { return __hfma2(a, b, c); }
};

__device__ __forceinline__ bool valid_level(int32_t height, int32_t width,
    int32_t start, int64_t spatial) {
    return height > 0 && width > 0 && start >= 0 && start < spatial
        && static_cast<int64_t>(height) * width <= spatial - start;
}

template <typename T> struct Sample {
    T x, y, weight;
};
template <typename T, typename Index>
__device__ __forceinline__ Sample<T> load_sample(const T* locations,
    const T* weights, Index index) {
    return {locations[2 * index], locations[2 * index + 1], weights[index]};
}

template <typename T, typename Index> struct Geometry {
    bool active;
    bool mask[4];
    Index offset[4];
    T coefficient[4];
};

template <typename T, typename Index>
__device__ __forceinline__ Geometry<T, Index> sample_geometry(int32_t height,
    int32_t width, T height_scalar, T width_scalar, Index pixel_stride,
    Sample<T> sample) {
    using S = Scalar<T>;
    Geometry<T, Index> geometry{};
    const T y = S::coordinate(sample.y, height_scalar);
    const T x = S::coordinate(sample.x, width_scalar);
    // Ordered comparisons also mask NaNs and infinities before conversion.
    if (!S::inside(y, height_scalar) || !S::inside(x, width_scalar)) return geometry;
    geometry.active = true;

    const int32_t y0 = S::floor_int(y), x0 = S::floor_int(x);
    const int32_t y1 = y0 + 1, x1 = x0 + 1;
    const T ly = S::sub(y, S::from_int(y0));
    const T lx = S::sub(x, S::from_int(x0));
    const T hy = S::sub(S::one(), ly), hx = S::sub(S::one(), lx);
    geometry.coefficient[0] = S::mul(hy, hx);
    geometry.coefficient[1] = S::mul(hy, lx);
    geometry.coefficient[2] = S::mul(ly, hx);
    geometry.coefficient[3] = S::mul(ly, lx);

    // Test both bounds for every neighbor: no speculative pointer is formed
    // for a masked pixel. These checks remain correct at -1 and at H/W.
    const bool m0 = y0 >= 0 && y0 < height && x0 >= 0 && x0 < width;
    const bool m1 = y0 >= 0 && y0 < height && x1 >= 0 && x1 < width;
    const bool m2 = y1 >= 0 && y1 < height && x0 >= 0 && x0 < width;
    const bool m3 = y1 >= 0 && y1 < height && x1 >= 0 && x1 < width;
    geometry.mask[0] = m0; geometry.mask[1] = m1;
    geometry.mask[2] = m2; geometry.mask[3] = m3;
    if constexpr (sizeof(Index) == 4) {
        // The host bounds every tensor byte span by INT32_MAX. The level was
        // checked with int64 arithmetic before this helper. Form only positive
        // row/column offsets, deriving adjacent neighbors by stride addition.
        const Index row_stride = static_cast<Index>(width) * pixel_stride;
        const Index r0 = y0 >= 0 && y0 < height ? static_cast<Index>(y0) * row_stride : 0;
        const Index r1 = y1 >= 0 && y1 < height ? (y0 >= 0 ? r0 + row_stride : 0) : 0;
        const Index c0 = x0 >= 0 && x0 < width ? static_cast<Index>(x0) * pixel_stride : 0;
        const Index c1 = x1 >= 0 && x1 < width ? (x0 >= 0 ? c0 + pixel_stride : 0) : 0;
        geometry.offset[0] = m0 ? r0 + c0 : 0;
        geometry.offset[1] = m1 ? r0 + c1 : 0;
        geometry.offset[2] = m2 ? r1 + c0 : 0;
        geometry.offset[3] = m3 ? r1 + c1 : 0;
    } else {
        geometry.offset[0] = m0 ? (static_cast<Index>(y0) * width + x0) * pixel_stride : 0;
        geometry.offset[1] = m1 ? (static_cast<Index>(y0) * width + x1) * pixel_stride : 0;
        geometry.offset[2] = m2 ? (static_cast<Index>(y1) * width + x0) * pixel_stride : 0;
        geometry.offset[3] = m3 ? (static_cast<Index>(y1) * width + x1) * pixel_stride : 0;
    }
    return geometry;
}

template <typename T, int Width>
__device__ __forceinline__ void accumulate_pack(const T (&coefficient)[4], T weight,
    typename Pack<T, Width>::Type v0, typename Pack<T, Width>::Type v1,
    typename Pack<T, Width>::Type v2, typename Pack<T, Width>::Type v3,
    typename Pack<T, Width>::Type& acc) {
    using V = Pack<T, Width>;
    using VType = typename V::Type;
    const VType w0 = V::broadcast(coefficient[0]), w1 = V::broadcast(coefficient[1]);
    const VType w2 = V::broadcast(coefficient[2]), w3 = V::broadcast(coefficient[3]);
    const VType attention = V::broadcast(weight);
    const VType b = V::mul(w1, v1), d = V::mul(w3, v3);
    if constexpr (std::is_same<T, __half>::value) {
        // Keep the second product rounded before each FMA. Fusing the
        // second instead of the first product gives different half bits.
        const VType interpolated = V::add(V::fma(w0, v0, b), V::fma(w2, v2, d));
        acc = V::fma(interpolated, attention, acc);
    } else {
        const VType a = V::mul(w0, v0), cterm = V::mul(w2, v2);
        const VType interpolated = V::add(V::add(V::add(a, b), cterm), d);
        acc = V::add(acc, V::mul(interpolated, attention));
    }
}

template <typename T, int Channels, int Width, typename Index>
__device__ __forceinline__ void accumulate_sample(const T* value,
    int32_t height, int32_t width, T height_scalar, T width_scalar,
    Index pixel_stride, Index first_channel, Index channels,
    Sample<T> sample, typename Pack<T, Width>::Type (&acc)[Channels / Width]) {
    using S = Scalar<T>;
    using V = Pack<T, Width>;
    using VType = typename V::Type;
    const T y = S::coordinate(sample.y, height_scalar);
    const T x = S::coordinate(sample.x, width_scalar);
    // Ordered comparisons also mask NaNs and infinities before conversion.
    if (!S::inside(y, height_scalar) || !S::inside(x, width_scalar)) return;

    const int32_t y0 = S::floor_int(y), x0 = S::floor_int(x);
    const int32_t y1 = y0 + 1, x1 = x0 + 1;
    const T ly = S::sub(y, S::from_int(y0));
    const T lx = S::sub(x, S::from_int(x0));
    const T hy = S::sub(S::one(), ly), hx = S::sub(S::one(), lx);
    // Keep the measured baseline's direct early return and broadcasts outside
    // the channel loop. Geometry/accumulate_pack are for prefetch 2 only.
    const VType w0 = V::broadcast(S::mul(hy, hx));
    const VType w1 = V::broadcast(S::mul(hy, lx));
    const VType w2 = V::broadcast(S::mul(ly, hx));
    const VType w3 = V::broadcast(S::mul(ly, lx));
    const VType attention = V::broadcast(sample.weight);
    const VType zero = V::broadcast(S::zero());

    // Test both bounds for every neighbor: no speculative pointer is formed
    // for a masked pixel. These checks remain correct at -1 and at H/W.
    const bool m0 = y0 >= 0 && y0 < height && x0 >= 0 && x0 < width;
    const bool m1 = y0 >= 0 && y0 < height && x1 >= 0 && x1 < width;
    const bool m2 = y1 >= 0 && y1 < height && x0 >= 0 && x0 < width;
    const bool m3 = y1 >= 0 && y1 < height && x1 >= 0 && x1 < width;
    Index o0, o1, o2, o3;
    if constexpr (sizeof(Index) == 4) {
        const Index row_stride = static_cast<Index>(width) * pixel_stride;
        const Index r0 = y0 >= 0 && y0 < height ? static_cast<Index>(y0) * row_stride : 0;
        const Index r1 = y1 >= 0 && y1 < height ? (y0 >= 0 ? r0 + row_stride : 0) : 0;
        const Index c0 = x0 >= 0 && x0 < width ? static_cast<Index>(x0) * pixel_stride : 0;
        const Index c1 = x1 >= 0 && x1 < width ? (x0 >= 0 ? c0 + pixel_stride : 0) : 0;
        o0 = m0 ? r0 + c0 : 0;
        o1 = m1 ? r0 + c1 : 0;
        o2 = m2 ? r1 + c0 : 0;
        o3 = m3 ? r1 + c1 : 0;
    } else {
        o0 = m0 ? (static_cast<Index>(y0) * width + x0) * pixel_stride : 0;
        o1 = m1 ? (static_cast<Index>(y0) * width + x1) * pixel_stride : 0;
        o2 = m2 ? (static_cast<Index>(y1) * width + x0) * pixel_stride : 0;
        o3 = m3 ? (static_cast<Index>(y1) * width + x1) * pixel_stride : 0;
    }
#pragma unroll
    for (int k = 0; k < Channels / Width; ++k) {
        const Index c = first_channel + k * Width;
        if (c >= channels) continue;
        const VType v0 = m0 ? V::load(value + o0 + c) : zero;
        const VType v1 = m1 ? V::load(value + o1 + c) : zero;
        const VType v2 = m2 ? V::load(value + o2 + c) : zero;
        const VType v3 = m3 ? V::load(value + o3 + c) : zero;
        const VType b = V::mul(w1, v1), d = V::mul(w3, v3);
        if (std::is_same<T, __half>::value) {
            // Keep the second product rounded before each FMA. Fusing the
            // second instead of the first product gives different half bits.
            const VType interpolated = V::add(V::fma(w0, v0, b), V::fma(w2, v2, d));
            acc[k] = V::fma(interpolated, attention, acc[k]);
        } else {
            // The float reference uses a left-associated four-term sum.
            const VType a = V::mul(w0, v0), cterm = V::mul(w2, v2);
            const VType interpolated = V::add(V::add(V::add(a, b), cterm), d);
            acc[k] = V::add(acc[k], V::mul(interpolated, attention));
        }
    }
}

template <typename T, int Channels, int Width> struct PreparedSample {
    bool active;
    T coefficient[4], weight;
    typename Pack<T, Width>::Type neighbor[4][Channels / Width];
};

template <typename T, int Channels, int Width, typename Index>
__device__ __forceinline__ PreparedSample<T, Channels, Width> prepare_sample(
    const T* value, int32_t height, int32_t width, T hs, T ws, Index pixel_stride,
    Index first_channel, Index channels, Sample<T> sample) {
    using V = Pack<T, Width>;
    PreparedSample<T, Channels, Width> prepared{};
    const auto geometry = sample_geometry<T, Index>(height, width, hs, ws, pixel_stride, sample);
    prepared.active = geometry.active;
    if (!geometry.active) return prepared;
    prepared.weight = sample.weight;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        prepared.coefficient[n] = geometry.coefficient[n];
#pragma unroll
        for (int k = 0; k < Channels / Width; ++k) {
            const Index c = first_channel + k * Width;
            prepared.neighbor[n][k] = geometry.mask[n] && c < channels
                ? V::load(value + geometry.offset[n] + c) : V::broadcast(Scalar<T>::zero());
        }
    }
    return prepared;
}

template <typename T, int Channels, int Width, typename Index>
__device__ __forceinline__ void accumulate_prepared(
    const PreparedSample<T, Channels, Width>& sample, Index first_channel, Index channels,
    typename Pack<T, Width>::Type (&acc)[Channels / Width]) {
    // Invalid coordinates must skip the attention operation, including NaN weights.
    if (!sample.active) return;
#pragma unroll
    for (int k = 0; k < Channels / Width; ++k) {
        if (first_channel + k * Width < channels) {
            accumulate_pack<T, Width>(sample.coefficient, sample.weight, sample.neighbor[0][k],
                sample.neighbor[1][k], sample.neighbor[2][k], sample.neighbor[3][k], acc[k]);
        }
    }
}

template <typename T, int Channels, int Width, bool Specialized, int Prefetch, typename Index>
__global__ __launch_bounds__(768) void msda_kernel(const T* __restrict__ value,
    const int32_t* __restrict__ shapes, const int32_t* __restrict__ starts,
    const T* __restrict__ locations, const T* __restrict__ weights,
    T* __restrict__ output, Dimensions dims, Index dynamic_groups, Index work) {
    using S = Scalar<T>;
    using V = Pack<T, Width>;
    using VType = typename V::Type;
    const Index channels = Specialized ? 32 : static_cast<Index>(dims.channels);
    const Index levels = Specialized ? 4 : static_cast<Index>(dims.levels);
    const Index points = Specialized ? 8 : static_cast<Index>(dims.points);
    const Index heads = static_cast<Index>(dims.heads), queries = static_cast<Index>(dims.queries);
    const Index spatial = static_cast<Index>(dims.spatial);
    const Index groups = Specialized ? 32 / Channels : dynamic_groups;
    const Index pixel_stride = heads * channels;
    const Index step = static_cast<Index>(blockDim.x) * gridDim.x;
    for (Index index = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < work; index += step) {
        const Index first_channel = (index % groups) * Channels;
        const Index sample_index = index / groups;
        const Index head = sample_index % heads;
        const Index batch = sample_index / heads / queries;
        const Index sample_base = sample_index * levels * points;
        const Index batch_base = batch * spatial * pixel_stride + head * channels;
        VType acc[Channels / Width];
#pragma unroll
        for (int k = 0; k < Channels / Width; ++k) acc[k] = V::broadcast(S::zero());

#pragma unroll 1
        for (Index level = 0; level < levels; ++level) {
            const int32_t height = shapes[2 * level], width = shapes[2 * level + 1];
            const int32_t start = starts[level];
            if (!valid_level(height, width, start, dims.spatial)) {
#pragma unroll
                for (int k = 0; k < Channels / Width; ++k) acc[k] = V::broadcast(S::nan());
                break;
            }
            const T hs = S::from_int(height), ws = S::from_int(width);
            const T* level_value = value + batch_base + static_cast<Index>(start) * pixel_stride;
            const Index point_base = sample_base + level * points;
            if constexpr (Prefetch == 2) {
                auto current = prepare_sample<T, Channels, Width>(level_value, height, width, hs, ws,
                    pixel_stride, first_channel, channels, load_sample(locations, weights, point_base));
                // One point ahead, with actual Value gathers issued before the
                // current interpolation. Do not unroll this register-heavy loop.
#pragma unroll 1
                for (Index point = 0; point < points; ++point) {
                    PreparedSample<T, Channels, Width> next{};
                    if (point + 1 < points) {
                        next = prepare_sample<T, Channels, Width>(level_value, height, width, hs, ws,
                            pixel_stride, first_channel, channels,
                            load_sample(locations, weights, point_base + point + 1));
                    }
                    accumulate_prepared<T, Channels, Width>(current, first_channel, channels, acc);
                    current = next;
                }
            } else {
                Sample<T> current{};
                if (Prefetch == 1) current = load_sample(locations, weights, point_base);
                // Preserve the original modest unrolling for prefetch 0/1.
#pragma unroll 2
                for (Index point = 0; point < points; ++point) {
                    Sample<T> next{};
                    if (Prefetch == 1) {
                        if (point + 1 < points) next = load_sample(locations, weights, point_base + point + 1);
                    } else {
                        current = load_sample(locations, weights, point_base + point);
                    }
                    accumulate_sample<T, Channels, Width>(level_value, height, width,
                        hs, ws, pixel_stride, first_channel, channels, current, acc);
                    if (Prefetch == 1) current = next;
                }
            }
        }
#pragma unroll
        for (int k = 0; k < Channels / Width; ++k) {
            const Index c = first_channel + k * Width;
            if (c < channels) V::store(output + sample_index * channels + c, acc[k]);
        }
    }
}

__global__ void validate_metadata_kernel(const int32_t* shapes,
    const int32_t* starts, int64_t spatial, int64_t levels, int32_t* status) {
    // A single warp avoids an initialization kernel or cross-block race.
    bool valid = true;
    for (int64_t level = threadIdx.x; level < levels; level += 32) {
        valid = valid && valid_level(shapes[2 * level], shapes[2 * level + 1], starts[level], spatial);
    }
    const bool all_valid = __all_sync(0xffffffffU, valid);
    if (threadIdx.x == 0) *status = all_valid ? GAC_MSDA_METADATA_OK : GAC_MSDA_METADATA_INVALID;
}

bool aligned(const void* pointer, size_t alignment) {
    return pointer != nullptr && reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

bool checked_product(int64_t a, int64_t b, int64_t& product) {
    if (a <= 0 || b <= 0 || a > std::numeric_limits<int64_t>::max() / b) return false;
    product = a * b;
    return true;
}

bool valid_dimensions(Dimensions d, size_t element_size) {
    if (d.batch <= 0 || d.spatial <= 0 || d.heads <= 0 || d.channels <= 0
        || d.levels <= 0 || d.queries <= 0 || d.points <= 0) return false;
    // Pointer arithmetic must fit ptrdiff_t in bytes, not just element counts.
    const uint64_t byte_limit = static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max());
    int64_t hc, values, samples, output, weights, locations, metadata;
    return checked_product(d.heads, d.channels, hc)
        && checked_product(d.batch, d.spatial, values)
        && checked_product(values, hc, values)
        && static_cast<uint64_t>(values) <= byte_limit / element_size
        && checked_product(d.batch, d.queries, samples)
        && checked_product(samples, d.heads, samples)
        && checked_product(samples, d.channels, output)
        && static_cast<uint64_t>(output) <= byte_limit / element_size
        && checked_product(samples, d.levels, weights)
        && checked_product(weights, d.points, weights)
        && checked_product(weights, 2, locations)
        && static_cast<uint64_t>(locations) <= byte_limit / element_size
        && checked_product(d.levels, 2, metadata)
        && static_cast<uint64_t>(metadata) <= byte_limit / sizeof(int32_t);
}

bool index32_safe(Dimensions d, size_t element_size) {
    // valid_dimensions has already established that these int64 products fit.
    // Bound bytes as well as elements, including both metadata tensors. This
    // also bounds every positive dimension and every derived element stride.
    const int64_t limit = std::numeric_limits<int32_t>::max();
    const int64_t tensor_limit = limit / static_cast<int64_t>(element_size);
    return d.batch * d.spatial * d.heads * d.channels <= tensor_limit
        && d.batch * d.queries * d.heads * d.channels <= tensor_limit
        && d.batch * d.queries * d.heads * d.levels * d.points * 2 <= tensor_limit
        && d.levels * 2 <= limit / static_cast<int64_t>(sizeof(int32_t));
}

template <typename T, int Channels, int Width, bool Specialized, int Prefetch, typename Index>
int launch(const void* value, const int32_t* shapes, const int32_t* starts,
    const void* locations, const void* weights, void* output, Dimensions dims,
    int block_threads, cudaStream_t stream) {
    const int64_t groups = dims.channels / Channels + (dims.channels % Channels != 0);
    const int64_t work = dims.batch * dims.queries * dims.heads * groups;
    int64_t blocks = work / block_threads + (work % block_threads != 0);
    // A portable grid bound plus a checked index type handles large inputs.
    if (blocks > 65535) blocks = 65535;
    msda_kernel<T, Channels, Width, Specialized, Prefetch, Index>
        <<<static_cast<unsigned int>(blocks), block_threads, 0, stream>>>(
            static_cast<const T*>(value), shapes, starts,
            static_cast<const T*>(locations), static_cast<const T*>(weights),
            static_cast<T*>(output), dims, static_cast<Index>(groups), static_cast<Index>(work));
    return static_cast<int>(cudaGetLastError());
}

template <typename T, int Channels, int Width, bool Specialized, typename Index>
int dispatch_prefetch(const void* value, const int32_t* shapes, const int32_t* starts,
    const void* locations, const void* weights, void* output, Dimensions dims,
    int block_threads, int prefetch, cudaStream_t stream) {
#define GAC_LAUNCH(PREFETCH) \
    launch<T, Channels, Width, Specialized, PREFETCH, Index>(value, shapes, starts, locations, \
        weights, output, dims, block_threads, stream)
    if (prefetch == 2) return GAC_LAUNCH(2);
    if (prefetch == 1) return GAC_LAUNCH(1);
    return GAC_LAUNCH(0);
#undef GAC_LAUNCH
}

template <typename T, int Channels, int Width, typename Index>
int dispatch_options(const void* value, const int32_t* shapes, const int32_t* starts,
    const void* locations, const void* weights, void* output, Dimensions dims,
    int block_threads, bool specialized, int prefetch, cudaStream_t stream) {
#define GAC_SPECIALIZED(SPECIAL) \
    dispatch_prefetch<T, Channels, Width, SPECIAL, Index>(value, shapes, starts, locations, \
        weights, output, dims, block_threads, prefetch, stream)
    if constexpr (sizeof(Index) == 4) {
        // The host only selects this index type for the fixed L4/P8/C32 shape.
        return GAC_SPECIALIZED(true);
    } else {
        if (specialized) return GAC_SPECIALIZED(true);
        return GAC_SPECIALIZED(false);
    }
#undef GAC_SPECIALIZED
}

template <typename T, typename Index>
int dispatch_channels(const void* value, const int32_t* shapes, const int32_t* starts,
    const void* locations, const void* weights, void* output, Dimensions dims,
    int channels_per_thread, int block_threads, bool specialized, int prefetch,
    cudaStream_t stream) {
#define GAC_CHANNELS(C) \
    dispatch_options<T, C, 1, Index>(value, shapes, starts, locations, weights, output, \
        dims, block_threads, specialized, prefetch, stream)
    if (channels_per_thread == 1) return GAC_CHANNELS(1);
    if (channels_per_thread == 2) return GAC_CHANNELS(2);
    return GAC_CHANNELS(4);
#undef GAC_CHANNELS
}

template <typename T, typename Index>
int dispatch_type(const void* value, const int32_t* shapes, const int32_t* starts,
    const void* locations, const void* weights, void* output, Dimensions dims,
    int channels_per_thread, int block_threads, bool specialized, int prefetch,
    cudaStream_t stream) {
    if constexpr (std::is_same<T, __half>::value) {
        // Even C and four-byte aligned value/output make all half2 gathers and
        // stores aligned. Other shapes and alignments retain scalar half math.
        if (dims.channels % 2 == 0 && aligned(value, alignof(__half2)) && aligned(output, alignof(__half2))) {
            if (channels_per_thread == 2) {
                return dispatch_options<__half, 2, 2, Index>(value, shapes, starts,
                    locations, weights, output, dims, block_threads, specialized, prefetch, stream);
            }
            if (channels_per_thread == 4) {
                return dispatch_options<__half, 4, 2, Index>(value, shapes, starts,
                    locations, weights, output, dims, block_threads, specialized, prefetch, stream);
            }
        }
    }
    return dispatch_channels<T, Index>(value, shapes, starts, locations, weights,
        output, dims, channels_per_thread, block_threads, specialized, prefetch, stream);
}

} // namespace

extern "C" GAC_MSDA_API int gac_msda_forward(const void* value,
    const int32_t* spatial_shapes, const int32_t* level_starts,
    const void* sampling_locations, const void* attention_weights, void* output,
    int64_t B, int64_t S, int64_t H, int64_t C, int64_t L, int64_t Q, int64_t P,
    int dtype, int channels_per_thread, int block_threads, int specialize,
    int prefetch, cudaStream_t stream) {
    const Dimensions dims{B, S, H, C, L, Q, P};
    const size_t element_size = dtype == GAC_MSDA_FLOAT32 ? sizeof(float) : sizeof(__half);
    if ((dtype != GAC_MSDA_FLOAT32 && dtype != GAC_MSDA_FLOAT16)
        || (channels_per_thread != 1 && channels_per_thread != 2 && channels_per_thread != 4)
        || (block_threads != 128 && block_threads != 256 && block_threads != 512 && block_threads != 768)
        || specialize < GAC_MSDA_GENERIC || specialize > GAC_MSDA_AUTO_INDEX32
        || (prefetch < 0 || prefetch > 2)
        || !valid_dimensions(dims, element_size)
        || !aligned(value, element_size) || !aligned(sampling_locations, element_size)
        || !aligned(attention_weights, element_size) || !aligned(output, element_size)
        || !aligned(spatial_shapes, alignof(int32_t)) || !aligned(level_starts, alignof(int32_t))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const bool matches = L == 4 && P == 8 && C == 32;
    if (specialize == GAC_MSDA_REQUIRE_L4_P8_C32 && !matches) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const bool specialized = specialize != GAC_MSDA_GENERIC && matches;
    const bool index32 = specialize == GAC_MSDA_AUTO_INDEX32 && matches && index32_safe(dims, element_size);
#define GAC_DISPATCH(T, INDEX) \
    dispatch_type<T, INDEX>(value, spatial_shapes, level_starts, sampling_locations, \
        attention_weights, output, dims, channels_per_thread, block_threads, specialized, prefetch, stream)
    if (dtype == GAC_MSDA_FLOAT32) {
        return index32 ? GAC_DISPATCH(float, uint32_t) : GAC_DISPATCH(float, int64_t);
    }
    return index32 ? GAC_DISPATCH(__half, uint32_t) : GAC_DISPATCH(__half, int64_t);
#undef GAC_DISPATCH
}

extern "C" GAC_MSDA_API int gac_msda_validate_metadata(const int32_t* spatial_shapes,
    const int32_t* level_starts, int64_t S, int64_t L,
    int32_t* device_status, cudaStream_t stream) {
    const int64_t max_levels = std::numeric_limits<ptrdiff_t>::max() / (2 * sizeof(int32_t));
    if (S <= 0 || L <= 0 || L > max_levels
        || !aligned(spatial_shapes, alignof(int32_t))
        || !aligned(level_starts, alignof(int32_t))
        || !aligned(device_status, alignof(int32_t))) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    validate_metadata_kernel<<<1, 32, 0, stream>>>(spatial_shapes, level_starts, S, L, device_status);
    return static_cast<int>(cudaGetLastError());
}
