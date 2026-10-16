cmake_minimum_required(VERSION 3.20)
project(GACNative LANGUAGES C CXX)

# Both device pipelines use one LLVM/MLIR release. Vortex additionally requires
# VOLT's RISC-V extensions; CUDA needs LLVM's standard NVPTX target.
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
if(NOT LLVM_PACKAGE_VERSION VERSION_EQUAL "20.1.8")
  message(FATAL_ERROR "GAC requires LLVM/MLIR 20.1.8 (VOLT fork for Vortex)")
endif()
find_package(Protobuf REQUIRED)
set(ONNX_PROTO_DIR "" CACHE PATH "Directory containing the official ONNX onnx.proto")
if(NOT EXISTS "${ONNX_PROTO_DIR}/onnx.proto")
  message(FATAL_ERROR "Set ONNX_PROTO_DIR to an installed ONNX package or ONNX source directory")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})

# Generate only the protobuf schema. The importer uses ModelProto directly;
# it does not link Python, build a Python Graph, or invoke Python passes.
protobuf_generate_cpp(ONNX_PROTO_SOURCES ONNX_PROTO_HEADERS "${ONNX_PROTO_DIR}/onnx.proto")
add_library(GACOnnxProto STATIC ${ONNX_PROTO_SOURCES})
target_link_libraries(GACOnnxProto PUBLIC protobuf::libprotobuf)

add_library(GACCompiler STATIC
  lib/Dialect.cpp
  lib/GraphPasses.cpp
  lib/ConstantFolding.cpp
  lib/TensorLegalization.cpp
  lib/OnnxImporter.cpp
  lib/DeviceLowering.cpp
  lib/CudaLowering.cpp
  lib/VortexLowering.cpp)
target_compile_options(GACCompiler PRIVATE -fno-rtti)
target_link_libraries(GACCompiler PUBLIC GACOnnxProto
  MLIRIR MLIRParser MLIRPass MLIRTransforms MLIRSupport MLIRSideEffectInterfaces
  MLIRFuncDialect MLIRArithDialect MLIRMathDialect MLIRSCFDialect
  MLIRControlFlowDialect MLIRLLVMDialect MLIRNVVMDialect
  MLIRSCFToControlFlow MLIRArithToLLVM MLIRMathToLLVM
  MLIRControlFlowToLLVM MLIRReconcileUnrealizedCasts)

add_executable(gac-compile tools/gac-compile.cpp)
target_compile_options(gac-compile PRIVATE -fno-rtti)
target_link_libraries(gac-compile PRIVATE GACCompiler)
add_executable(gac-opt tools/gac-opt.cpp)
target_compile_options(gac-opt PRIVATE -fno-rtti)
target_link_libraries(gac-opt PRIVATE GACCompiler MLIROptLib)
install(TARGETS gac-compile gac-opt RUNTIME DESTINATION bin)

# CUDA is optional: only NVIDIA's official headers are required to build the
# runtime. libcuda is dynamically loaded on the execution host, not linked.
set(GAC_CUDA_INCLUDE_DIR "" CACHE PATH "Directory containing official cuda.h")
if(GAC_CUDA_INCLUDE_DIR)
  if(NOT EXISTS "${GAC_CUDA_INCLUDE_DIR}/cuda.h")
    message(FATAL_ERROR "GAC_CUDA_INCLUDE_DIR must contain NVIDIA cuda.h")
  endif()
  add_executable(gac-cuda-run runtime/cuda-run.cpp)
  target_compile_options(gac-cuda-run PRIVATE -fno-rtti)
  target_include_directories(gac-cuda-run PRIVATE ${GAC_CUDA_INCLUDE_DIR})
  target_link_libraries(gac-cuda-run PRIVATE LLVMSupport ${CMAKE_DL_LIBS})
  install(TARGETS gac-cuda-run RUNTIME DESTINATION bin)
endif()

set(VORTEX_BUILD "" CACHE PATH "Configured Vortex build directory (optional host runtime)")
set(VORTEX_SOURCE "" CACHE PATH "Vortex source tree containing sw/runtime/include")
if(VORTEX_BUILD)
  find_path(VORTEX_INCLUDE vortex2.h
    PATHS "${VORTEX_SOURCE}/sw/runtime/include" "$ENV{VORTEX_HOME}/sw/runtime/include"
          "${VORTEX_BUILD}/sw/runtime/include" "${VORTEX_BUILD}/../sw/runtime/include"
    REQUIRED NO_DEFAULT_PATH)
  find_library(VORTEX_RUNTIME NAMES vortex
    PATHS "${VORTEX_BUILD}/sw/runtime" REQUIRED NO_DEFAULT_PATH)
  add_executable(gac-vortex-run runtime/vortex-run.cpp)
  target_compile_options(gac-vortex-run PRIVATE -fno-rtti)
  target_include_directories(gac-vortex-run PRIVATE ${VORTEX_INCLUDE})
  target_link_libraries(gac-vortex-run PRIVATE LLVMSupport ${VORTEX_RUNTIME})
  install(TARGETS gac-vortex-run RUNTIME DESTINATION bin)
endif()
