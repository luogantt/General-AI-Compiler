# Use the configured Vortex ABI, startup, linker script and kernel library.
# ROOT_DIR and LLVM_PATH must be passed on the make command line.
ifndef ROOT_DIR
$(error pass ROOT_DIR=/absolute/path/to/configured/vortex/build)
endif
include $(ROOT_DIR)/config.mk
ifneq ($(XLEN),32)
$(error native GAC currently targets the RV32 Vortex ABI)
endif
PROJECT := gac_native_vortex
VX_SRCS := kernel.ll
KERNEL_LIB := vortex2
# common.mk also supports building a host executable. Our host runtime is built
# by CMake, so provide an empty dependency file instead of invoking g++ with no
# host source files when common.mk remakes its included .depend.
ifeq ($(wildcard .depend),)
$(file >.depend,# Host dependencies are managed by native/CMakeLists.txt.)
endif
include $(ROOT_DIR)/tests/regression/common.mk
VX_CFLAGS += -ffixed-x31
VX_LDFLAGS += -fuse-ld=lld --ld-path=$(LLVM_PATH)/bin/ld.lld

.PHONY: check-native-toolchain
check-native-toolchain:
	$(LLVM_PATH)/bin/mlir-translate --version | grep -F '20.1.8' >/dev/null
	$(LLVM_PATH)/bin/clang --version | grep -F '20.1.8' >/dev/null
	test -r kernel.mlir

kernel.ll: kernel.mlir | check-native-toolchain
	$(LLVM_PATH)/bin/mlir-translate --mlir-to-llvmir $< -o $@

vx_start.o kernel.elf kernel.vxbin: | check-native-toolchain
