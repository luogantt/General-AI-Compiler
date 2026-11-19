// Inference-only eager bridge to the separately built GacGdnDecode operator.
// Uses TorchNPU's exported OpCommand/task-queue path and the current NPU stream.
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

#include "aclnn_gac_gdn_decode.h"
#include "acl/acl.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace gac_qwen38 {
namespace {

constexpr int64_t HK = 16;
constexpr int64_t HV = 48;
constexpr int64_t DK = 128;
constexpr int64_t DV = 128;

// Check the generated ABI rather than assuming the Float attribute's C type.
template <typename FunctionPointer>
struct ApiArguments;

template <typename Result, typename... Arguments>
struct ApiArguments<Result (*)(Arguments...)> {
  using Tuple = std::tuple<Arguments...>;
  static constexpr std::size_t count = sizeof...(Arguments);
};

using WorkspaceApi = ApiArguments<decltype(&aclnnGacGdnDecodeGetWorkspaceSize)>;
static_assert(WorkspaceApi::count == 11,
              "Expected q, k, v, g, beta, state, l2_eps, o, state_out, "
              "workspace_size, executor");
static_assert(std::is_same_v<std::tuple_element_t<6, WorkspaceApi::Tuple>, double>,
              "Generated l2_eps ABI must be double; do not guess the ACLNN ABI");

void check_input(const at::Tensor& tensor, const char* name,
                 const c10::Device& device, at::ScalarType dtype) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              name, " must be on the NPU PrivateUse1 backend");
  TORCH_CHECK(tensor.device() == device, name, " must be on the same NPU as q");
  TORCH_CHECK(tensor.layout() == c10::kStrided, name, " must have strided layout");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous; no implicit copy is performed");
  const auto format = at_npu::native::custom_ops::get_npu_format(tensor);
  TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW,
              name, " must use unblocked ND/NCHW format; no implicit format cast is performed");
}

// Keep both the ATen storage references and ACL descriptors alive until the
// queued launch has completed on the host.
struct AclTensorOwner {
  at::Tensor tensor;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  aclTensor* descriptor = nullptr;

  AclTensorOwner(const at::Tensor& value, aclDataType dtype)
      : tensor(value), sizes(value.sizes().vec()), strides(value.strides().vec()) {
    descriptor = aclCreateTensor(sizes.data(), sizes.size(), dtype,
                                 strides.data(), 0, ACL_FORMAT_ND,
                                 sizes.data(), sizes.size(), tensor.data_ptr());
    TORCH_CHECK(descriptor != nullptr, "aclCreateTensor failed for GacGdnDecode");
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
  AclTensorOwner q, k, v, g, beta, state, o, state_out;
  double l2_eps;
  Invocation(const at::Tensor& qv, const at::Tensor& kv, const at::Tensor& vv,
             const at::Tensor& gv, const at::Tensor& betav,
             const at::Tensor& statev, const at::Tensor& ov,
             const at::Tensor& stateOutv, double eps)
      : q(qv, ACL_BF16), k(kv, ACL_BF16), v(vv, ACL_BF16), g(gv, ACL_FLOAT),
        beta(betav, ACL_FLOAT), state(statev, ACL_FLOAT), o(ov, ACL_BF16),
        state_out(stateOutv, ACL_FLOAT), l2_eps(eps) {}
};

void check_acl_status(aclnnStatus status, const char* operation) {
  if (status != ACL_SUCCESS) {
    const char* detail = aclGetRecentErrMsg();
    TORCH_CHECK(false, operation, " failed (", status, "): ",
                detail ? detail : "no ACL error detail");
  }
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> gdn_decode(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const at::Tensor& g, const at::Tensor& beta, const at::Tensor& state,
    double l2_eps) {
  TORCH_CHECK(!c10::GradMode::is_enabled(),
              "gac_qwen38::gdn_decode is inference-only; use no_grad/inference_mode");
  check_input(q, "q", q.device(), at::kBFloat16);
  check_input(k, "k", q.device(), at::kBFloat16);
  check_input(v, "v", q.device(), at::kBFloat16);
  check_input(g, "g", q.device(), at::kFloat);
  check_input(beta, "beta", q.device(), at::kFloat);
  check_input(state, "state", q.device(), at::kFloat);

  TORCH_CHECK(q.dim() == 2 && q.size(0) == HK && q.size(1) == DK,
              "Expected BF16 q[16,128]");
  TORCH_CHECK(k.sizes() == q.sizes(), "Expected BF16 k[16,128]");
  TORCH_CHECK(v.dim() == 2 && v.size(0) == HV && v.size(1) == DV,
              "Expected BF16 v[48,128]");
  TORCH_CHECK(g.dim() == 1 && g.size(0) == HV, "Expected FP32 g[48]");
  TORCH_CHECK(beta.sizes() == g.sizes(), "Expected FP32 beta[48]");
  TORCH_CHECK(state.dim() == 3 && state.size(0) == HV && state.size(1) == DK &&
                  state.size(2) == DV,
              "Expected FP32 state[48,128,128]");
  const float eps_fp32 = static_cast<float>(l2_eps);
  TORCH_CHECK(std::isfinite(l2_eps) && l2_eps > 0.0 &&
                  std::isfinite(eps_fp32) && eps_fp32 > 0.0f,
              "l2_eps must remain finite and positive in FP32");

  const c10::DeviceGuard device_guard(q.device());
  auto output = at::empty({HV, DV}, q.options());
  auto state_out = at::empty({HV, DK, DV}, state.options());
  check_input(output, "output", q.device(), at::kBFloat16);
  check_input(state_out, "state_out", q.device(), at::kFloat);

  const auto stream = c10_npu::getCurrentNPUStream(q.device().index()).stream(false);
  auto invocation = std::make_shared<Invocation>(q, k, v, g, beta, state, output,
                                                 state_out, l2_eps);
  // Queue BOTH executor preparation and launch behind prior TorchNPU commands.
  // This callback calls no ATen allocator/operator. The audited tiling requests
  // zero workspace; reject a different contract rather than borrowing
  // temporary memory with an unsafe asynchronous lifetime.
  at_npu::native::OpCommand::RunOpApi("aclnnGacGdnDecode", [invocation, stream]() -> int {
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnGacGdnDecodeGetWorkspaceSize(
        invocation->q.descriptor, invocation->k.descriptor,
        invocation->v.descriptor, invocation->g.descriptor,
        invocation->beta.descriptor, invocation->state.descriptor,
        invocation->l2_eps, invocation->o.descriptor,
        invocation->state_out.descriptor, &workspace_size, &executor);
    check_acl_status(status, "aclnnGacGdnDecodeGetWorkspaceSize");
    TORCH_CHECK(workspace_size == 0,
                "GacGdnDecode bridge requires zero workspace; received ", workspace_size);
    TORCH_CHECK(executor != nullptr, "GacGdnDecode returned a null executor");
    status = aclnnGacGdnDecode(nullptr, 0, executor, stream);
    check_acl_status(status, "aclnnGacGdnDecode");
    return static_cast<int>(status);
  }, false);
  return {output, state_out};
}

}  // namespace gac_qwen38

TORCH_LIBRARY_FRAGMENT(gac_qwen38, library) {
  library.def("gdn_decode(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
              "Tensor state, float l2_eps) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gac_qwen38, PrivateUse1, library) {
  library.impl("gdn_decode", TORCH_FN(gac_qwen38::gdn_decode));
}
