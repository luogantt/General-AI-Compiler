cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures for DCN im2col")
endif()
project(gac_dcn_im2col LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
add_library(gac_dcn_im2col SHARED dcn_im2col.cu)
target_link_libraries(gac_dcn_im2col PRIVATE CUDA::cudart)
target_compile_features(gac_dcn_im2col PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(gac_dcn_im2col PRIVATE GAC_DCN_BUILD)
# Upstream half intrinsics permit contraction. Do not substitute the MSDA
# project's --fmad=false policy or enable --use_fast_math for this experiment.
target_compile_options(gac_dcn_im2col PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>)
set_target_properties(gac_dcn_im2col PROPERTIES
  CXX_EXTENSIONS OFF CUDA_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF WINDOWS_EXPORT_ALL_SYMBOLS ON)
install(TARGETS gac_dcn_im2col LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(FILES dcn_im2col.h DESTINATION include/gac_dcn)

# Keep the small independently testable kernel build free of TensorRT/cuBLAS
# requirements unless the opt-in full DCNv2 plugin is requested.
option(GAC_BUILD_DCN_PLUGIN "Build the FP16 TensorRT DCNv2 plugin (validated static subset)" OFF)
if(GAC_BUILD_DCN_PLUGIN)
  set(TENSORRT_ROOT "" CACHE PATH "Optional TensorRT SDK root")
  set(TENSORRT_INCLUDE_DIR "" CACHE PATH "Directory containing TensorRT 10.13 headers")
  set(TENSORRT_LIBRARY_DIR "" CACHE PATH "Directory containing pip libnvinfer.so.10")
  set(TENSORRT_LIBRARY "" CACHE FILEPATH "Explicit TensorRT nvinfer library")
  set(CUBLAS_INCLUDE_DIR "" CACHE PATH "Directory containing cublas_v2.h (including pip nvidia/cublas/include)")
  set(CUBLAS_LIBRARY_DIR "" CACHE PATH "Directory containing pip nvidia/cublas/lib")
  set(CUBLAS_LIBRARY "" CACHE FILEPATH "Explicit cuBLAS library, including versioned libcublas.so.12")
  if(NOT TENSORRT_INCLUDE_DIR)
    find_path(_dcn_trt_include NvInferRuntime.h HINTS "${TENSORRT_ROOT}" "$ENV{TENSORRT_ROOT}"
      PATH_SUFFIXES include include/x86_64-linux-gnu)
    set(TENSORRT_INCLUDE_DIR "${_dcn_trt_include}" CACHE PATH "TensorRT headers" FORCE)
  endif()
  if(NOT TENSORRT_LIBRARY)
    find_library(_dcn_trt_library NAMES nvinfer nvinfer_10
      HINTS "${TENSORRT_LIBRARY_DIR}" "${TENSORRT_ROOT}" "$ENV{TENSORRT_ROOT}"
      PATH_SUFFIXES lib lib64 tensorrt_libs tensorrt_bindings lib/x86_64-linux-gnu)
    if(NOT _dcn_trt_library)
      find_file(_dcn_trt_versioned NAMES libnvinfer.so.10
        HINTS "${TENSORRT_LIBRARY_DIR}" "${TENSORRT_ROOT}" "$ENV{TENSORRT_ROOT}"
        PATH_SUFFIXES lib lib64 tensorrt_libs tensorrt_bindings lib/x86_64-linux-gnu)
      set(_dcn_trt_library "${_dcn_trt_versioned}")
    endif()
    set(TENSORRT_LIBRARY "${_dcn_trt_library}" CACHE FILEPATH "TensorRT library" FORCE)
  endif()
  if(NOT CUBLAS_INCLUDE_DIR)
    find_path(_dcn_cublas_include cublas_v2.h HINTS ${CUDAToolkit_INCLUDE_DIRS}
      "${CUDAToolkit_TARGET_DIR}" "$ENV{CUBLAS_ROOT}" PATH_SUFFIXES include nvidia/cublas/include)
    set(CUBLAS_INCLUDE_DIR "${_dcn_cublas_include}" CACHE PATH "cuBLAS headers" FORCE)
  endif()
  if(NOT CUBLAS_LIBRARY)
    find_library(_dcn_cublas_library NAMES cublas cublas64_12
      HINTS "${CUBLAS_LIBRARY_DIR}" "${CUDAToolkit_LIBRARY_DIR}" "$ENV{CUBLAS_ROOT}"
      PATH_SUFFIXES lib lib64 nvidia/cublas/lib)
    if(NOT _dcn_cublas_library)
      find_file(_dcn_cublas_versioned NAMES libcublas.so.12
        HINTS "${CUBLAS_LIBRARY_DIR}" "${CUDAToolkit_LIBRARY_DIR}" "$ENV{CUBLAS_ROOT}"
        PATH_SUFFIXES lib lib64 nvidia/cublas/lib)
      set(_dcn_cublas_library "${_dcn_cublas_versioned}")
    endif()
    set(CUBLAS_LIBRARY "${_dcn_cublas_library}" CACHE FILEPATH "cuBLAS library" FORCE)
  endif()
  if(NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferRuntime.h" OR NOT EXISTS "${TENSORRT_LIBRARY}")
    message(FATAL_ERROR "DCN plugin requires TENSORRT_INCLUDE_DIR and TENSORRT_LIBRARY (TensorRT 10.13).")
  endif()
  if(NOT EXISTS "${CUBLAS_INCLUDE_DIR}/cublas_v2.h" OR NOT EXISTS "${CUBLAS_LIBRARY}")
    message(FATAL_ERROR "Set CUBLAS_INCLUDE_DIR and CUBLAS_LIBRARY; pip nvidia/cublas/include and lib layouts are supported.")
  endif()
  find_package(Threads REQUIRED)
  add_library(gac_dcn_plugin SHARED plugin.cpp dcn_im2col.cu)
  target_include_directories(gac_dcn_plugin PRIVATE "${TENSORRT_INCLUDE_DIR}" "${CUBLAS_INCLUDE_DIR}")
  target_link_libraries(gac_dcn_plugin PRIVATE "${TENSORRT_LIBRARY}" "${CUBLAS_LIBRARY}"
    CUDA::cudart CUDA::cuda_driver Threads::Threads)
  target_compile_features(gac_dcn_plugin PRIVATE cxx_std_17 cuda_std_17)
  target_compile_definitions(gac_dcn_plugin PRIVATE GAC_DCN_BUILD)
  target_compile_options(gac_dcn_plugin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>)
  set_target_properties(gac_dcn_plugin PROPERTIES CXX_EXTENSIONS OFF CUDA_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON CUDA_SEPARABLE_COMPILATION OFF WINDOWS_EXPORT_ALL_SYMBOLS ON)
  get_filename_component(_dcn_trt_libdir "${TENSORRT_LIBRARY}" DIRECTORY)
  get_filename_component(_dcn_cublas_libdir "${CUBLAS_LIBRARY}" DIRECTORY)
  set_property(TARGET gac_dcn_plugin PROPERTY BUILD_RPATH
    "${_dcn_trt_libdir};${_dcn_cublas_libdir};${CUDAToolkit_LIBRARY_DIR}")
  set_property(TARGET gac_dcn_plugin PROPERTY INSTALL_RPATH "$ORIGIN;${_dcn_trt_libdir};${_dcn_cublas_libdir}")
  install(TARGETS gac_dcn_plugin LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
  message(STATUS "GACModulatedDeformConv2d: TensorRT=${TENSORRT_LIBRARY}; cuBLAS=${CUBLAS_LIBRARY}")
endif()
