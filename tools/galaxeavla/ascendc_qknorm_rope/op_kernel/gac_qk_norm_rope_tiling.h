#ifndef GAC_QK_NORM_ROPE_TILING_H
#define GAC_QK_NORM_ROPE_TILING_H

#include <cstdint>

// Fixed LIBERO AE contract. Keep this POD identical on host and device.
struct GacQkNormRopeTilingData {
    uint32_t headCount;
    uint32_t headStride;
    uint32_t projectionWidth;
    uint32_t rowCount;
    float epsilon;
};

#endif
