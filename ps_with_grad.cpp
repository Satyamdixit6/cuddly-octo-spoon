#include <torch/extension.h>
#include <vector>

at::Tensor vector_add_forward(at::Tensor a, at::Tensor b); // Forward declaration

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> vector_add_backward(at::Tensor grad_output) {
    CHECK_INPUT(grad_output);
    // Simple backward: gradient of sum is 1 w.r.t each input
    return {grad_output, grad_output};
}

class VectorAddFunction : public torch::autograd::Function<VectorAddFunction> {
public:
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        at::Tensor a,
        at::Tensor b)
    {
        return vector_add_forward(a, b);
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        auto grad_output = grad_outputs[0];
        auto grads = vector_add_backward(grad_output);
        return {grads[0], grads[1]};
    }
};

at::Tensor vector_add_autograd(at::Tensor a, at::Tensor b) {
  return VectorAddFunction::apply(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add_autograd, "Vector Addition with Autograd (CUDA)");
}