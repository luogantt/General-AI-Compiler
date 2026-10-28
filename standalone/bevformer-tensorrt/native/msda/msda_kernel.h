// SPDX-License-Identifier: Apache-2.0
#ifndef GAC_MSDA_KERNEL_H
#define GAC_MSDA_KERNEL_H

#include <stdint.h>
#include <cuda_runtime_api.h>

#if defined(_WIN32)
#if defined(GAC_MSDA_BUILD)
#define GAC_MSDA_API __declspec(dllexport)
#else
#define GAC_MSDA_API
#endif
#else
#define GAC_MSDA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum GacMsdaDtype { GAC_MSDA_FLOAT32 = 0, GAC_MSDA_FLOAT16 = 1 };
enum GacMsdaSpecialization {
    GAC_MSDA_GENERIC = 0,
    GAC_MSDA_AUTO = 1,
    GAC_MSDA_REQUIRE_L4_P8_C32 = 2,
    GAC_MSDA_AUTO_INDEX32 = 3
};
enum GacMsdaMetadataStatus { GAC_MSDA_METADATA_OK = 0, GAC_MSDA_METADATA_INVALID = 1 };

// All tensors are contiguous, device-resident, and non-overlapping:
// value[B,S,H,C], spatial_shapes[L,2] (height,width), level_starts[L],
// sampling_locations[B,Q,H,L,P,2] (x,y), attention_weights[B,Q,H,L,P],
// output[B,Q,H,C]. Shapes and starts are int32; the other tensors use dtype.
//
// channels_per_thread: 1, 2, or 4; block_threads: 128, 256, 512, or 768.
// prefetch: 0 off; 1 loads the next point's location/weight; 2 also gathers
// the next point's four Value neighbors before accumulating the current point.
// Specialization AUTO uses the L=4/P=8/C=32 kernel only when all three match;
// REQUIRE rejects other shapes. Odd C and channel tails are supported.
// AUTO_INDEX32 uses uint32 element indices and adjacent-neighbor stride reuse
// for L=4/P=8/C=32 only when every tensor byte span fits INT32_MAX; otherwise it
// falls back to the original 64-bit AUTO path. Host overflow checks and device
// metadata validation still use 64-bit arithmetic. Existing defaults do not change.
//
// Returns a cudaError_t value as int. Invalid pointers, nonpositive dimensions,
// unsupported launch parameters, and dimension/address overflow return
// cudaErrorInvalidValue before launch. Allocation sizes and pointer residency
// are the caller's responsibility. No allocation or synchronization is done;
// work is enqueued on stream and CUDA execution errors remain asynchronous.
// Requires an SM >= 53 target for native FP16 arithmetic. Do not use fast math.
// FP16 uses explicit half FMA at the coordinates, first product of each
// interpolation pair, and attention accumulation, matching the measured
// NVIDIA TensorRT 10.13.3 SM86 binary. Other rounding boundaries stay separate;
// point order is unchanged. Revalidate this contract on other binary versions.
//
// Device metadata cannot be reported synchronously without a stream barrier.
// A nonpositive level size, negative start, or level extending past S causes
// output NaNs, without reading values or locations for that invalid level.
// Use gac_msda_validate_metadata to obtain an explicit asynchronous status.
// In-bounds levels need not be contiguous, exhaustive, or disjoint.
GAC_MSDA_API int gac_msda_forward(const void* value, const int32_t* spatial_shapes,
    const int32_t* level_starts, const void* sampling_locations,
    const void* attention_weights, void* output, int64_t B, int64_t S,
    int64_t H, int64_t C, int64_t L, int64_t Q, int64_t P, int dtype,
    int channels_per_thread, int block_threads, int specialize, int prefetch,
    cudaStream_t stream);

// Writes exactly one GacMsdaMetadataStatus to device_status on stream.
// The caller reads that value after its own stream/event synchronization.
// Status validates each int32 [height,width,start] against positive S and L.
// A successful host return means the validator was launched, not that the
// metadata is valid. Host argument errors return cudaErrorInvalidValue.
GAC_MSDA_API int gac_msda_validate_metadata(const int32_t* spatial_shapes,
    const int32_t* level_starts, int64_t S, int64_t L,
    int32_t* device_status, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif
