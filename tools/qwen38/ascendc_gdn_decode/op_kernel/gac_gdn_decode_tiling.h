#ifndef GAC_GDN_DECODE_TILING_H
#define GAC_GDN_DECODE_TILING_H

#include <cstdint>

// Fixed Qwen3.8-27B decode contract (batch 1, sequence length 1).
// Keep this POD identical on host and device.
struct GacGdnDecodeTilingData {
    uint32_t headCount;   // value heads (48)
    uint32_t headGroup;   // value heads per key head (3)
    float l2Eps;          // 1e-6, matches FLA l2norm
};

#endif
