# Reusable compute templates depend on target primitives, never on the ONNX
# importer, graph passes, compiler driver or Runtime. This keeps links acyclic.
add_library(GACComputeTemplates STATIC
  Registry.cpp
  cuda/Matmul.cpp
  cuda/Reduction.cpp
  cuda/Convolution.cpp
  vortex/Matmul.cpp
  vortex/Tcu.cpp)
target_compile_options(GACComputeTemplates PRIVATE -fno-rtti)
target_link_libraries(GACComputeTemplates PUBLIC GACTargetSupport
  MLIRIR MLIRSupport MLIRArithDialect MLIRMathDialect MLIRSCFDialect
  MLIRLLVMDialect MLIRNVVMDialect)

add_executable(gac-template-list ../tools/gac-template-list.cpp)
target_compile_options(gac-template-list PRIVATE -fno-rtti)
target_link_libraries(gac-template-list PRIVATE GACComputeTemplates)
install(TARGETS gac-template-list RUNTIME DESTINATION bin)
