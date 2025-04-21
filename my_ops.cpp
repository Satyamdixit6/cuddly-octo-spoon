#include <torch/extension.h>
#include <vector>

at::Tensor vector_add_forward_cuda(at::Tensor a, at::Tensor b);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

at::Tensor vector_add_forward(at::Tensor a, at::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same size");
    TORCH_CHECK(a.scalar_type() == at::kFloat, "Input tensors must be float32");
    TORCH_CHECK(b.scalar_type() == at::kFloat, "Input tensors must be float32");

    return vector_add_forward_cuda(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add_forward, "Vector Addition (CUDA)");
}