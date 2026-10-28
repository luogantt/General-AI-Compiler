cmake_minimum_required(VERSION 3.24)

# Set this before enabling CUDA so a caller's -DCMAKE_CUDA_ARCHITECTURES wins.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures for GAC MSDA")
endif()
project(gac_msda LANGUAGES CXX CUDA)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT SDK root (optional)")
set(TENSORRT_INCLUDE_DIR "" CACHE PATH "Directory containing TensorRT 10.13 NvInferRuntime.h")
set(TENSORRT_LIBRARY_DIR "" CACHE PATH "Directory containing libnvinfer (for example pip tensorrt_libs)")
set(TENSORRT_LIBRARY "" CACHE FILEPATH "Explicit libnvinfer shared/import library path")

find_package(CUDAToolkit REQUIRED)
if(NOT TENSORRT_INCLUDE_DIR)
  find_path(_gac_trt_include NvInferRuntime.h
    HINTS "${TENSORRT_ROOT}" "$ENV{TENSORRT_ROOT}"
    PATH_SUFFIXES include include/x86_64-linux-gnu)
  set(TENSORRT_INCLUDE_DIR "${_gac_trt_include}" CACHE PATH "TensorRT include directory" FORCE)
endif()
if(NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferRuntime.h")
  message(FATAL_ERROR "Set TENSORRT_INCLUDE_DIR to the TensorRT 10.13 header directory. A pip runtime wheel may not include headers.")
endif()

# Wheel layouts commonly contain only libnvinfer.so.10, without the development
# libnvinfer.so symlink. Both a directory and an explicit file are supported.
if(NOT TENSORRT_LIBRARY)
  find_library(_gac_trt_library NAMES nvinfer nvinfer_10
    HINTS "${TENSORRT_LIBRARY_DIR}" "${TENSORRT_ROOT}" "$ENV{TENSORRT_ROOT}"
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu tensorrt_libs tensorrt_bindings)
  if(NOT _gac_trt_library)
    find_file(_gac_trt_library_versioned NAMES libnvinfer.so.10
      HINTS "${TENSORRT_LIBRARY_DIR}" "${TENSORRT_ROOT}" "$ENV{TENSORRT_ROOT}"
      PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu tensorrt_libs tensorrt_bindings)
    set(_gac_trt_library "${_gac_trt_library_versioned}")
  endif()
  set(TENSORRT_LIBRARY "${_gac_trt_library}" CACHE FILEPATH "TensorRT nvinfer library" FORCE)
endif()
if(NOT EXISTS "${TENSORRT_LIBRARY}")
  message(FATAL_ERROR "Set TENSORRT_LIBRARY_DIR to pip tensorrt_libs, or TENSORRT_LIBRARY to the complete nvinfer library filename.")
endif()

add_library(gac_msda_plugin SHARED plugin.cpp msda_kernel.cu)
target_include_directories(gac_msda_plugin PRIVATE "${TENSORRT_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(gac_msda_plugin PRIVATE "${TENSORRT_LIBRARY}" CUDA::cudart)
target_compile_features(gac_msda_plugin PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(gac_msda_plugin PRIVATE GAC_MSDA_BUILD)
target_compile_options(gac_msda_plugin PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(gac_msda_plugin PROPERTIES
  CXX_EXTENSIONS OFF
  CUDA_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION OFF
  WINDOWS_EXPORT_ALL_SYMBOLS ON)
# msda_kernel.h marks its C entry points with default visibility (or dllexport).
# Keep the shared object, including the directly testable C kernel API, loaded
# for the lifetime of any TensorRT engine using its registered creator.
get_filename_component(_gac_trt_libdir "${TENSORRT_LIBRARY}" DIRECTORY)
set_property(TARGET gac_msda_plugin PROPERTY BUILD_RPATH "${_gac_trt_libdir};${CUDAToolkit_LIBRARY_DIR}")
set_property(TARGET gac_msda_plugin PROPERTY INSTALL_RPATH "$ORIGIN;${_gac_trt_libdir}")

install(TARGETS gac_msda_plugin LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(FILES msda_kernel.h DESTINATION include/gac_msda)
message(STATUS "GACMSDeformAttn: TensorRT headers=${TENSORRT_INCLUDE_DIR}")
message(STATUS "GACMSDeformAttn: TensorRT library=${TENSORRT_LIBRARY}")
message(STATUS "GACMSDeformAttn: CUDA architectures=${CMAKE_CUDA_ARCHITECTURES}; fmad=false")
