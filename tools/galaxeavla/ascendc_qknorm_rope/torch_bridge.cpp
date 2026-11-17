// Inference-only eager bridge to the separately built AscendC custom operator.
// Uses TorchNPU's exported OpCommand/task-queue path and current NPU stream.
// No host/device synchronization, CPU copy, global stream, or fallback kernel.
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/GradMode.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

#include "aclnn_gac_qk_norm_rope.h"
#include "acl/acl.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace gac_g05 {
namespace {

// Check the generated ABI rather than assuming the Float attribute's C type.
template <typename FunctionPointer>
struct ApiArguments;

template <typename Result, typename... Arguments>
struct ApiArguments<Result (*)(Arguments...)> {
  using Tuple = std::tuple<Arguments...>;
  static constexpr std::size_t count = sizeof...(Arguments);
};

using WorkspaceApi = ApiArguments<decltype(&aclnnGacQkNormRopeGetWorkspaceSize)>;
static_assert(WorkspaceApi::count == 9,
              "Expected x, weight, cos, sin, head_stride, epsilon, y, workspace_size, executor");
static_assert(std::is_same_v<std::tuple_element_t<4, WorkspaceApi::Tuple>, int64_t>,
              "Generated head_stride ABI must be int64_t");
static_assert(std::is_same_v<std::tuple_element_t<5, WorkspaceApi::Tuple>, double>,
              "Generated epsilon ABI must be double; do not guess the ACLNN ABI");

void check_input(const at::Tensor& tensor, const char* name,
                 const c10::Device& device, at::ScalarType dtype) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              name, " must be on the NPU PrivateUse1 backend");
  TORCH_CHECK(tensor.device() == device, name, " must be on the same NPU as x");
  TORCH_CHECK(tensor.layout() == c10::kStrided, name, " must have strided layout");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous; no implicit copy is performed");
  const auto format = at_npu::native::custom_ops::get_npu_format(tensor);
  // TorchNPU uses NCHW for some rank-four tensors even with internal formats
  // disabled. It is the same unblocked dense storage for these fixed shapes.
  TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW,
              name, " must use unblocked ND/NCHW format; no implicit format cast is performed");
}

// Keep both the ATen storage references and ACL descriptors alive until the
// queued launch has completed on the host. ACL owns the executor after launch;
// descriptor destruction does not free the ATen-owned device buffers.
struct AclTensorOwner {
  at::Tensor tensor;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  aclTensor* descriptor = nullptr;

  AclTensorOwner(const at::Tensor& value, aclDataType dtype)
      : tensor(value), sizes(value.sizes().vec()), strides(value.strides().vec()) {
    // data_ptr already includes a view's storage_offset. Describe only this
    // contiguous logical extent with offset zero; never expose preceding data.
    descriptor = aclCreateTensor(sizes.data(), sizes.size(), dtype,
                                 strides.data(), 0, ACL_FORMAT_ND,
                                 sizes.data(), sizes.size(), tensor.data_ptr());
    TORCH_CHECK(descriptor != nullptr, "aclCreateTensor failed for GacQkNormRope");
  }
  AclTensorOwner(const AclTensorOwner&) = delete;
  AclTensorOwner& operator=(const AclTensorOwner&) = delete;
  ~AclTensorOwner() {
    if (descriptor != nullptr) {
      (void)aclDestroyTensor(descriptor);
    }
  }
};

struct Invocation {
  AclTensorOwner x, weight, cos, sin, output;
  int64_t head_stride;
  double epsilon;
  Invocation(const at::Tensor& x_value, const at::Tensor& weight_value,
             const at::Tensor& cos_value, const at::Tensor& sin_value,
             const at::Tensor& output_value, int64_t stride, double eps)
      : x(x_value, ACL_BF16), weight(weight_value, ACL_FLOAT),
        cos(cos_value, ACL_FLOAT), sin(sin_value, ACL_FLOAT),
        output(output_value, ACL_BF16), head_stride(stride), epsilon(eps) {}
};

void check_acl_status(aclnnStatus status, const char* operation) {
  if (status != ACL_SUCCESS) {
    const char* detail = aclGetRecentErrMsg();
    TORCH_CHECK(false, operation, " failed (", status, "): ", detail ? detail : "no ACL error detail");
  }
}

}  // namespace

at::Tensor qk_norm_rope(const at::Tensor& x, const at::Tensor& weight,
                       const at::Tensor& cos, const at::Tensor& sin,
                       int64_t head_stride, double epsilon) {
  TORCH_CHECK(!c10::GradMode::is_enabled(),
              "gac_g05::qk_norm_rope is inference-only; use no_grad/inference_mode");
  TORCH_CHECK(x.defined(), "x must be defined");
  check_input(x, "x", x.device(), at::kBFloat16);
  check_input(weight, "weight", x.device(), at::kFloat);
  check_input(cos, "cos", x.device(), at::kFloat);
  check_input(sin, "sin", x.device(), at::kFloat);
  TORCH_CHECK(head_stride == 512 || head_stride == 256,
              "head_stride must be 512 for Q or 256 for K");
  const int64_t heads = head_stride == 512 ? 8 : 2;
  const int64_t storage_width = head_stride == 512 ? 4096 : 512;
  TORCH_CHECK(x.dim() == 3 && x.size(0) == 1 && x.size(1) == 32 &&
                  x.size(2) == storage_width,
              "Expected contiguous BF16 x[1,32,4096] for Q or x[1,32,512] for K");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == 256,
              "Expected FP32 weight[256]");
  TORCH_CHECK(cos.dim() == 4 && cos.size(0) == 1 && cos.size(1) == 1 &&
                  cos.size(2) >= 32 && cos.size(3) == 64,
              "Expected FP32 cos[1,1,S>=32,64]");
  TORCH_CHECK(sin.sizes() == cos.sizes(), "sin shape must exactly match cos");
  const float epsilon_fp32 = static_cast<float>(epsilon);
  TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0 &&
                  std::isfinite(epsilon_fp32) && epsilon_fp32 > 0.0f,
              "epsilon must remain finite and positive when represented in FP32");

  // DeviceGuard changes the current device, not the caller's stream selection.
  const c10::DeviceGuard device_guard(x.device());
  auto output = at::empty({1, heads, 32, 256}, x.options());
  check_input(output, "output", x.device(), at::kBFloat16);
  const auto stream = c10_npu::getCurrentNPUStream(x.device().index()).stream(false);
  auto invocation = std::make_shared<Invocation>(x, weight, cos, sin, output, head_stride, epsilon);
  // Queue BOTH executor preparation and launch behind prior TorchNPU commands.
  // This callback calls no ATen allocator/operator, so it cannot recursively
  // enqueue a PyTorch allocation while running on the command-queue worker.
  // The audited tiling requests zero workspace; reject a different contract
  // rather than borrowing temporary memory with an unsafe asynchronous lifetime.
  at_npu::native::OpCommand::RunOpApi("aclnnGacQkNormRope", [invocation, stream]() -> int {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnGacQkNormRopeGetWorkspaceSize(
        invocation->x.descriptor, invocation->weight.descriptor,
        invocation->cos.descriptor, invocation->sin.descriptor,
        invocation->head_stride, invocation->epsilon, invocation->output.descriptor,
        &workspace_size, &executor);
    check_acl_status(status, "aclnnGacQkNormRopeGetWorkspaceSize");
    TORCH_CHECK(workspace_size == 0,
                "GacQkNormRope bridge requires zero workspace; received ", workspace_size);
    TORCH_CHECK(executor != nullptr, "GacQkNormRope returned a null executor");
    status = aclnnGacQkNormRope(nullptr, 0, executor, stream);
    check_acl_status(status, "aclnnGacQkNormRope");
    return static_cast<int>(status);
  }, false);
  return output;
}

}  // namespace gac_g05

TORCH_LIBRARY_FRAGMENT(gac_g05, library) {
  library.def("qk_norm_rope(Tensor x, Tensor weight, Tensor cos, Tensor sin, "
              "int head_stride, float epsilon) -> Tensor");
}

TORCH_LIBRARY_IMPL(gac_g05, PrivateUse1, library) {
  library.impl("qk_norm_rope", TORCH_FN(gac_g05::qk_norm_rope));
}
