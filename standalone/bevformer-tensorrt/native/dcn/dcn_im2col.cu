/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * Bilinear arithmetic and im2col layout adapted from NVIDIA TensorRT
 * release/10.13/plugin/modulatedDeformConvPlugin/modulatedDeformConvPluginKernel.cu
 * https://github.com/NVIDIA/TensorRT/tree/release/10.13/plugin/modulatedDeformConvPlugin
 * Upstream modified from MMCV, Copyright (c) OpenMMLab. All Rights Reserved.
 * https://github.com/open-mmlab/mmcv/blob/master/LICENSE (Apache-2.0).
 * GAC changes: checked C ABI, explicit nonfinite/full neighbor guards,
 * launch candidates, and independently selected 3x3 loop specialization.
 */

#ifndef GAC_DCN_BUILD
#define GAC_DCN_BUILD
#endif
#include "dcn_im2col.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <type_traits>

namespace {

struct Dimensions {
    int32_t batch, channels, height, width, out_h, out_w, kernel_h, kernel_w;
    int32_t pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, deform_groups;
};

template <typename T>
__device__ __forceinline__ T bilinear(const T* input, int32_t height,
    int32_t width, float h, float w) {
    // Ordered comparisons mask NaN/Inf before floor or pointer arithmetic.
    if (!(h > -1.0F && h < height && w > -1.0F && w < width)) return T(0);
    const int32_t y0 = static_cast<int32_t>(floorf(h));
    const int32_t x0 = static_cast<int32_t>(floorf(w));
    const int32_t y1 = y0 + 1, x1 = x0 + 1;
    const T ly = h - y0, lx = w - x0;
    const T hy = 1 - ly, hx = 1 - lx;
    T v0 = 0, v1 = 0, v2 = 0, v3 = 0;
    if (y0 >= 0 && y0 < height && x0 >= 0 && x0 < width) v0 = input[y0 * width + x0];
    if (y0 >= 0 && y0 < height && x1 >= 0 && x1 < width) v1 = input[y0 * width + x1];
    if (y1 >= 0 && y1 < height && x0 >= 0 && x0 < width) v2 = input[y1 * width + x0];
    if (y1 >= 0 && y1 < height && x1 >= 0 && x1 < width) v3 = input[y1 * width + x1];
    const T w0 = hy * hx, w1 = hy * lx, w2 = ly * hx, w3 = ly * lx;
    return w0 * v0 + w1 * v1 + w2 * v2 + w3 * v3;
}

template <>
__device__ __forceinline__ __half bilinear(const __half* input, int32_t height,
    int32_t width, float h, float w) {
    if (!(h > -1.0F && h < height && w > -1.0F && w < width)) return __float2half(0);
    const int32_t y0 = static_cast<int32_t>(floorf(h));
    const int32_t x0 = static_cast<int32_t>(floorf(w));
    const int32_t y1 = y0 + 1, x1 = x0 + 1;
    const __half ly = __float2half(h - y0), lx = __float2half(w - x0);
    const __half hy = __float2half(1) - ly, hx = __float2half(1) - lx;
    __half v0 = 0, v1 = 0, v2 = 0, v3 = 0;
    if (y0 >= 0 && y0 < height && x0 >= 0 && x0 < width) v0 = input[y0 * width + x0];
    if (y0 >= 0 && y0 < height && x1 >= 0 && x1 < width) v1 = input[y0 * width + x1];
    if (y1 >= 0 && y1 < height && x0 >= 0 && x0 < width) v2 = input[y1 * width + x0];
    if (y1 >= 0 && y1 < height && x1 >= 0 && x1 < width) v3 = input[y1 * width + x1];
    const __half w0 = __hmul(hy, hx), w1 = __hmul(hy, lx);
    const __half w2 = __hmul(ly, hx), w3 = __hmul(ly, lx);
    // Preserve upstream's half pairwise tree and contraction-capable intrinsics.
    return __hadd(__hadd(__hmul(w0, v0), __hmul(w1, v1)),
        __hadd(__hmul(w2, v2), __hmul(w3, v3)));
}

template <typename T>
__device__ __forceinline__ void sample_point(const T* input,
    const T* offset, const T* mask, T* column, Dimensions d,
    int32_t row, int32_t col, int32_t kh, int32_t kw, int32_t point,
    int32_t spatial, int32_t out_hw, int32_t column_stride) {
    const T dy = offset[(2 * point) * out_hw + spatial];
    const T dx = offset[(2 * point + 1) * out_hw + spatial];
    const T modulation = mask[point * out_hw + spatial];
    // Deliberately convert back to T here, as the NVIDIA source does. Keeping
    // these coordinates in float for FP16 would change sampling and rounding.
    const T y = row + kh * d.dilation_h + static_cast<float>(dy);
    const T x = col + kw * d.dilation_w + static_cast<float>(dx);
    const T value = bilinear(input, d.height, d.width, static_cast<float>(y), static_cast<float>(x));
    column[point * column_stride] = value * modulation;
}

template <typename T, bool Kernel3x3>
__global__ __launch_bounds__(1024) void im2col_kernel(const T* __restrict__ input,
    const T* __restrict__ offset, const T* __restrict__ mask,
    T* __restrict__ columns, Dimensions d, int32_t work) {
    // Host work <= INT32_MAX, including the last block's unsigned tail lanes.
    const uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= static_cast<uint32_t>(work)) return;
    const int32_t index = static_cast<int32_t>(linear);
    const int32_t out_hw = d.out_h * d.out_w;
    const int32_t spatial = index % out_hw;
    const int32_t bc = index / out_hw;
    const int32_t batch = bc % d.batch, channel = bc / d.batch;
    const int32_t group = channel / (d.channels / d.deform_groups);
    const int32_t kernel_h = Kernel3x3 ? 3 : d.kernel_h;
    const int32_t kernel_w = Kernel3x3 ? 3 : d.kernel_w;
    const int32_t kernel_hw = kernel_h * kernel_w;
    const int32_t row = (spatial / d.out_w) * d.stride_h - d.pad_h;
    const int32_t col = (spatial % d.out_w) * d.stride_w - d.pad_w;
    const T* image = input + (batch * d.channels + channel) * d.height * d.width;
    const int32_t group_base = (batch * d.deform_groups + group) * kernel_hw * out_hw;
    const T* group_offset = offset + 2 * group_base;
    const T* group_mask = mask + group_base;
    T* column = columns + (channel * kernel_hw * d.batch + batch) * out_hw + spatial;
    const int32_t column_stride = d.batch * out_hw;
    if constexpr (Kernel3x3) {
#pragma unroll
        for (int32_t kh = 0; kh < 3; ++kh) {
#pragma unroll
            for (int32_t kw = 0; kw < 3; ++kw) {
                sample_point(image, group_offset, group_mask, column, d,
                    row, col, kh, kw, kh * 3 + kw, spatial, out_hw, column_stride);
            }
        }
    } else {
#pragma unroll 1
        for (int32_t kh = 0; kh < kernel_h; ++kh) {
#pragma unroll 1
            for (int32_t kw = 0; kw < kernel_w; ++kw) {
                sample_point(image, group_offset, group_mask, column, d,
                    row, col, kh, kw, kh * kernel_w + kw, spatial, out_hw, column_stride);
            }
        }
    }
}

template <typename T, int Channels>
__global__ __launch_bounds__(1024) void im2col_channels_kernel(const T* __restrict__ input,
    const T* __restrict__ offset, const T* __restrict__ mask,
    T* __restrict__ columns, Dimensions d, int32_t work) {
    const uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= static_cast<uint32_t>(work)) return;
    const int32_t index = static_cast<int32_t>(linear);
    const int32_t out_hw = d.out_h * d.out_w;
    const int32_t spatial = index % out_hw;
    const int32_t bc = index / out_hw;
    const int32_t batch = bc % d.batch, channel_group = bc / d.batch;
    const int32_t channels_per_group = d.channels / d.deform_groups;
    const int32_t groups_per_deform = channels_per_group / Channels + (channels_per_group % Channels != 0);
    const int32_t deform_group = channel_group / groups_per_deform;
    const int32_t local_channel = (channel_group % groups_per_deform) * Channels;
    const int32_t first_channel = deform_group * channels_per_group + local_channel;
    const int32_t row = (spatial / d.out_w) * d.stride_h - d.pad_h;
    const int32_t col = (spatial % d.out_w) * d.stride_w - d.pad_w;
    const int32_t image_hw = d.height * d.width;
    const T* image = input + (batch * d.channels + first_channel) * image_hw;
    const int32_t group_base = (batch * d.deform_groups + deform_group) * 9 * out_hw;
    const T* group_offset = offset + 2 * group_base;
    const T* group_mask = mask + group_base;
    const int32_t column_stride = d.batch * out_hw;
    T* column = columns + (first_channel * 9 * d.batch + batch) * out_hw + spatial;

    // Keep neighboring lanes on neighboring spatial positions for NCHW loads.
    // Channels belong to one deformable group even when its channel count has
    // a tail. The original one-channel kernels above are deliberately separate.
#pragma unroll
    for (int32_t kh = 0; kh < 3; ++kh) {
#pragma unroll
        for (int32_t kw = 0; kw < 3; ++kw) {
            const int32_t point = kh * 3 + kw;
            const T dy = group_offset[(2 * point) * out_hw + spatial];
            const T dx = group_offset[(2 * point + 1) * out_hw + spatial];
            const T modulation = group_mask[point * out_hw + spatial];
            const T y = row + kh * d.dilation_h + static_cast<float>(dy);
            const T x = col + kw * d.dilation_w + static_cast<float>(dx);
            const float h = static_cast<float>(y), w = static_cast<float>(x);
            if (!(h > -1.0F && h < d.height && w > -1.0F && w < d.width)) {
                // Preserve zero * mask even for a nonfinite or negative mask.
                const T value = T(0) * modulation;
#pragma unroll
                for (int k = 0; k < Channels; ++k) {
                    if (local_channel + k < channels_per_group) {
                        column[(k * 9 + point) * column_stride] = value;
                    }
                }
                continue;
            }
            const int32_t y0 = static_cast<int32_t>(floorf(h));
            const int32_t x0 = static_cast<int32_t>(floorf(w));
            const int32_t y1 = y0 + 1, x1 = x0 + 1;
            T w0, w1, w2, w3;
            if constexpr (std::is_same<T, __half>::value) {
                const __half ly = __float2half(h - y0), lx = __float2half(w - x0);
                const __half hy = __float2half(1) - ly, hx = __float2half(1) - lx;
                w0 = __hmul(hy, hx); w1 = __hmul(hy, lx);
                w2 = __hmul(ly, hx); w3 = __hmul(ly, lx);
            } else {
                const T ly = h - y0, lx = w - x0;
                const T hy = 1 - ly, hx = 1 - lx;
                w0 = hy * hx; w1 = hy * lx; w2 = ly * hx; w3 = ly * lx;
            }
            const bool m0 = y0 >= 0 && y0 < d.height && x0 >= 0 && x0 < d.width;
            const bool m1 = y0 >= 0 && y0 < d.height && x1 >= 0 && x1 < d.width;
            const bool m2 = y1 >= 0 && y1 < d.height && x0 >= 0 && x0 < d.width;
            const bool m3 = y1 >= 0 && y1 < d.height && x1 >= 0 && x1 < d.width;
            const int32_t o0 = m0 ? y0 * d.width + x0 : 0;
            const int32_t o1 = m1 ? y0 * d.width + x1 : 0;
            const int32_t o2 = m2 ? y1 * d.width + x0 : 0;
            const int32_t o3 = m3 ? y1 * d.width + x1 : 0;
#pragma unroll
            for (int k = 0; k < Channels; ++k) {
                if (local_channel + k >= channels_per_group) continue;
                const T* channel_image = image + k * image_hw;
                const T v0 = m0 ? channel_image[o0] : T(0);
                const T v1 = m1 ? channel_image[o1] : T(0);
                const T v2 = m2 ? channel_image[o2] : T(0);
                const T v3 = m3 ? channel_image[o3] : T(0);
                T sampled;
                if constexpr (std::is_same<T, __half>::value) {
                    sampled = __hadd(__hadd(__hmul(w0, v0), __hmul(w1, v1)),
                        __hadd(__hmul(w2, v2), __hmul(w3, v3)));
                } else {
                    sampled = w0 * v0 + w1 * v1 + w2 * v2 + w3 * v3;
                }
                column[(k * 9 + point) * column_stride] = sampled * modulation;
            }
        }
    }
}

bool aligned(const void* pointer, size_t alignment) {
    return pointer != nullptr && reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

bool positive_i32(int64_t n) {
    return n > 0 && n <= std::numeric_limits<int32_t>::max();
}

bool bounded_product(std::initializer_list<int64_t> factors, size_t element_size) {
    const int64_t limit = std::numeric_limits<int32_t>::max();
    int64_t count = 1;
    for (const int64_t n : factors) {
        if (!positive_i32(n) || count > limit / n) return false;
        count *= n;
    }
    return static_cast<uint64_t>(count) <=
        static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max()) / element_size;
}

bool valid_axis(int64_t input, int64_t output, int64_t kernel, int64_t pad,
    int64_t stride, int64_t dilation) {
    const int64_t limit = std::numeric_limits<int32_t>::max();
    if (!positive_i32(input) || !positive_i32(output) || !positive_i32(kernel)
        || !positive_i32(stride) || !positive_i32(dilation) || pad < 0 || pad > limit) return false;
    // Products of two validated int32 positives fit int64; check before any
    // int32 coordinate intermediates in the device code are formed.
    const int64_t extent = dilation * (kernel - 1);
    const int64_t last_origin = (output - 1) * stride;
    const int64_t numerator = input + 2 * pad - extent - 1;
    return extent <= limit && last_origin <= limit && numerator >= 0
        && numerator / stride + 1 == output
        && last_origin - pad + extent <= limit;
}

template <typename T, bool Kernel3x3>
int launch(const void* input, const void* offset, const void* mask, void* columns,
    Dimensions d, int32_t work, int block, cudaStream_t stream) {
    const uint32_t grid = static_cast<uint32_t>((static_cast<int64_t>(work) + block - 1) / block);
    im2col_kernel<T, Kernel3x3><<<grid, block, 0, stream>>>(
        static_cast<const T*>(input), static_cast<const T*>(offset),
        static_cast<const T*>(mask), static_cast<T*>(columns), d, work);
    return static_cast<int>(cudaGetLastError());
}

template <typename T, int Channels>
int launch_channels(const void* input, const void* offset, const void* mask,
    void* columns, Dimensions d, int block, cudaStream_t stream) {
    const int32_t channels_per_group = d.channels / d.deform_groups;
    const int32_t groups_per_deform = channels_per_group / Channels + (channels_per_group % Channels != 0);
    // This work count is <= the already-checked one-channel work count.
    const int32_t work = static_cast<int32_t>(static_cast<int64_t>(d.batch)
        * d.deform_groups * groups_per_deform * d.out_h * d.out_w);
    const uint32_t grid = static_cast<uint32_t>((static_cast<int64_t>(work) + block - 1) / block);
    im2col_channels_kernel<T, Channels><<<grid, block, 0, stream>>>(
        static_cast<const T*>(input), static_cast<const T*>(offset),
        static_cast<const T*>(mask), static_cast<T*>(columns), d, work);
    return static_cast<int>(cudaGetLastError());
}

} // namespace

extern "C" GAC_DCN_API int dcn_im2col(const void* input, const void* offset,
    const void* mask, void* columns, int64_t batch, int64_t C, int64_t H,
    int64_t W, int64_t Hout, int64_t Wout, int64_t kH, int64_t kW,
    int64_t padH, int64_t padW, int64_t strideH, int64_t strideW,
    int64_t dilationH, int64_t dilationW, int64_t deformGroups,
    int dtype, int block, int specialize, cudaStream_t stream) {
    const size_t element_size = dtype == GAC_DCN_FLOAT32 ? sizeof(float) : sizeof(__half);
    if ((dtype != GAC_DCN_FLOAT32 && dtype != GAC_DCN_FLOAT16)
        || (block != 128 && block != 256 && block != 512 && block != 1024)
        || specialize < GAC_DCN_GENERIC || specialize > GAC_DCN_AUTO_3X3_C4
        || !positive_i32(batch) || !positive_i32(C) || !positive_i32(deformGroups)
        || C % deformGroups != 0
        || !valid_axis(H, Hout, kH, padH, strideH, dilationH)
        || !valid_axis(W, Wout, kW, padW, strideW, dilationW)
        || !bounded_product({batch, C, H, W}, element_size)
        || !bounded_product({batch, deformGroups, 2, kH, kW, Hout, Wout}, element_size)
        || !bounded_product({C, kH, kW, batch, Hout, Wout}, element_size)
        || !aligned(input, element_size) || !aligned(offset, element_size)
        || !aligned(mask, element_size) || !aligned(columns, element_size)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const bool matches = kH == 3 && kW == 3;
    if (specialize == GAC_DCN_REQUIRE_3X3 && !matches) return static_cast<int>(cudaErrorInvalidValue);
    const bool specialized = specialize != GAC_DCN_GENERIC && matches;
    const Dimensions d{static_cast<int32_t>(batch), static_cast<int32_t>(C),
        static_cast<int32_t>(H), static_cast<int32_t>(W), static_cast<int32_t>(Hout), static_cast<int32_t>(Wout),
        static_cast<int32_t>(kH), static_cast<int32_t>(kW), static_cast<int32_t>(padH), static_cast<int32_t>(padW),
        static_cast<int32_t>(strideH), static_cast<int32_t>(strideW), static_cast<int32_t>(dilationH),
        static_cast<int32_t>(dilationW), static_cast<int32_t>(deformGroups)};
    const int32_t work = static_cast<int32_t>(batch * C * Hout * Wout);
#define GAC_DCN_CHANNELS(T, CHANNELS) \
    launch_channels<T, CHANNELS>(input, offset, mask, columns, d, block, stream)
    if (matches && specialize == GAC_DCN_AUTO_3X3_C2) {
        return dtype == GAC_DCN_FLOAT32 ? GAC_DCN_CHANNELS(float, 2) : GAC_DCN_CHANNELS(__half, 2);
    }
    if (matches && specialize == GAC_DCN_AUTO_3X3_C4) {
        return dtype == GAC_DCN_FLOAT32 ? GAC_DCN_CHANNELS(float, 4) : GAC_DCN_CHANNELS(__half, 4);
    }
#undef GAC_DCN_CHANNELS
#define GAC_DCN_LAUNCH(T, SPECIALIZED) \
    launch<T, SPECIALIZED>(input, offset, mask, columns, d, work, block, stream)
    if (dtype == GAC_DCN_FLOAT32) {
        return specialized ? GAC_DCN_LAUNCH(float, true) : GAC_DCN_LAUNCH(float, false);
    }
    return specialized ? GAC_DCN_LAUNCH(__half, true) : GAC_DCN_LAUNCH(__half, false);
#undef GAC_DCN_LAUNCH
}
