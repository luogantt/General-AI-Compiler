#ifndef GAC_DEVICE_OPTIONS_H
#define GAC_DEVICE_OPTIONS_H

#include <cstdint>
#include <string>

namespace gac {
enum class DeviceTarget { Vortex, Native, CUDA };

struct DeviceOptions {
  // Native is an explicit numerical test target, never an execution fallback.
  DeviceTarget target = DeviceTarget::Vortex;
  std::string constantsDirectory;
  // Explicit target schedules keep the scalar baseline reproducible. Shapes
  // outside a template's contract retain scalar code with a manifest reason.
  std::string vortexMatmulSchedule = "scalar";
  int64_t tileM = 4, tileN = 4, tileK = 16;
  // LMEM/DXA output elements per thread; 1x1 preserves the original template.
  int64_t vortexThreadTileM = 1, vortexThreadTileN = 1;
  bool vortexTcuCombined = false;
  int64_t vortexLanes = 4;
  int64_t lmemBudget = 16384;
  std::string cudaMatmulSchedule = "scalar";
  int64_t cudaTileM = 32, cudaTileN = 32, cudaTileK = 16;
  int64_t cudaThreadTileM = 2, cudaThreadTileN = 2;
  std::string cudaReductionSchedule = "scalar";
  int64_t cudaConvChannelTile = 1;
  // Regular tensor algebra uses Linalg by default. The old device-specific
  // templates remain an explicit baseline; they must never bypass this choice.
  std::string regularLowering = "linalg";
  // Optional TTIR artifacts for eligible structured pointwise dispatches.
  // The common allocator and dependency plan remain authoritative.
  std::string vortexTritonDirectory;
  int64_t tritonBlock = 128, tritonNumWarps = 1;
};

} // namespace gac
#endif
