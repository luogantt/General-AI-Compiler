// SPDX-License-Identifier: Apache-2.0
#ifndef GAC_DCN_IM2COL_H
#define GAC_DCN_IM2COL_H

#include <stdint.h>
#include <cuda_runtime_api.h>

#if defined(_WIN32) && defined(GAC_DCN_BUILD)
#define GAC_DCN_API __declspec(dllexport)
#elif defined(_WIN32)
#define GAC_DCN_API
#else
#define GAC_DCN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum GacDcnDtype { GAC_DCN_FLOAT32 = 0, GAC_DCN_FLOAT16 = 1 };
enum GacDcnSpecialization {
    GAC_DCN_GENERIC = 0,
    GAC_DCN_AUTO_3X3 = 1,
    GAC_DCN_REQUIRE_3X3 = 2,
    GAC_DCN_AUTO_3X3_C2 = 3,
    GAC_DCN_AUTO_3X3_C4 = 4
};

// Contiguous, device-resident, non-overlapping buffers of the same dtype:
// input   [batch,C,H,W]
// offset  [batch,2*deformGroups*kH*kW,Hout,Wout]
// mask    [batch,deformGroups*kH*kW,Hout,Wout]
// columns [C*kH*kW,batch,Hout,Wout]
// Offset channels are interleaved (dy,dx) for each row-major kernel position.
// Mask is consumed as supplied; this function does not apply sigmoid.
// The columns layout is NVIDIA's im2col layout, NOT [batch,C*kH*kW,Hout*Wout].
// The built-in TensorRT DCN launches im2col separately for each batch item;
// batch=1 therefore reproduces its per-image GEMM input layout directly.
//
// All dimension arguments are int64_t at the ABI; validated device arithmetic
// uses int32_t. Dimensions, element counts and coordinate intermediates must
// fit INT32_MAX; byte spans must also fit ptrdiff_t. Empty shapes, C not divisible
// by deformGroups, inconsistent convolution output sizes, unsupported options,
// null/misaligned pointers and overflow return cudaErrorInvalidValue prelaunch.
// The caller owns buffer sizes, residency and lifetimes. Stride/dilation must
// be positive and padding nonnegative. General rectangular kernels are valid.
//
// block: 128,256,512,1024. specialize: 0 generic; 1 unrolled 3x3 when kH=kW=3,
// otherwise generic; 2 requires kH=kW=3. Padding/stride/dilation remain runtime
// values in the 3x3 specialization. No model-specific weights are used.
// Modes 3/4 use 2/4 channels per thread for 3x3, otherwise the generic path.
// Coordinate/coefficient/offset/mask reuse stays within each deformable group;
// channels per deformable group may have tails. Existing modes are unchanged.
//
// FP16 follows TensorRT release/10.13: integer base + float(offset) is converted
// back to half BEFORE bilinear sampling; fractional parts and interpolation
// weights use half; the pairwise half sum is multiplied by the half mask.
// Native half intrinsics retain upstream compiler contraction behavior; this
// source contract must be checked against the installed TensorRT binary.
// Nonfinite coordinates yield zero sampling, avoiding invalid pointer reads.
// No GEMM, bias, allocation, copy, synchronization, or precision promotion of
// interpolation is performed. Enqueues on stream; returns cudaError_t as int.
// A successful return is launch success; execution errors remain asynchronous.
GAC_DCN_API int dcn_im2col(const void* input, const void* offset,
    const void* mask, void* columns, int64_t batch, int64_t C, int64_t H,
    int64_t W, int64_t Hout, int64_t Wout, int64_t kH, int64_t kW,
    int64_t padH, int64_t padW, int64_t strideH, int64_t strideW,
    int64_t dilationH, int64_t dilationW, int64_t deformGroups,
    int dtype, int block, int specialize, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif
