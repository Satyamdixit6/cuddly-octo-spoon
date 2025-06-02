#include <torch/extension.h>
#include <vector>
#include <cuda_runtime.h>

// CUDA forward declaration - ensure this matches your .cu file
extern "C" cudaError_t conv3d_forward_cuda_launcher(
    const float* input,      // Input tensor (N, C, D_in, H_in, W_in)
    const float* weights,    // Filter tensor (K, C, D_f, H_f, W_f)
    float* output,           // Output tensor (N, K, D_out, H_out, W_out)
    const float* bias,       // Bias tensor (K) - can be nullptr
    int N, int C, int D_in, int H_in, int W_in,
    int K, int D_f, int H_f, int W_f,
    int D_out, int H_out, int W_out,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor conv3d_forward_cpp(
    torch::Tensor input,     // N, C, D_in, H_in, W_in
    torch::Tensor weights,   // K, C, D_f, H_f, W_f
    torch::Tensor bias,      // K (or empty if no bias)
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w) {

    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    // Bias tensor will be checked more carefully below if it has elements

    TORCH_CHECK(input.dim() == 5, "Input tensor must be 5D (NCDHW)");
    TORCH_CHECK(weights.dim() == 5, "Weights tensor must be 5D (KCDHW)");

    const int N = input.size(0);
    const int C = input.size(1);
    const int D_in = input.size(2);
    const int H_in = input.size(3);
    const int W_in = input.size(4);

    const int K = weights.size(0);
    TORCH_CHECK(weights.size(1) == C, "weights.size(1) (channels) must match input.size(1)");
    const int D_f = weights.size(2);
    const int H_f = weights.size(3);
    const int W_f = weights.size(4);

    const int D_out = (D_in + 2 * pad_d - D_f) / stride_d + 1;
    const int H_out = (H_in + 2 * pad_h - H_f) / stride_h + 1;
    const int W_out = (W_in + 2 * pad_w - W_f) / stride_w + 1;

    TORCH_CHECK(D_out > 0 && H_out > 0 && W_out > 0, "Output dimensions must be positive.");

    auto output = torch::zeros({N, K, D_out, H_out, W_out}, input.options());

    const float* bias_ptr_for_kernel = nullptr; // Initialize to nullptr

    if (bias.defined()) { // Check if a bias tensor was passed (even if empty)
        if (bias.numel() > 0) { // If it has elements, it's a real bias
            CHECK_INPUT(bias); // Must be CUDA & contiguous
            TORCH_CHECK(bias.dim() == 1, "Provided bias must be 1D");
            TORCH_CHECK(bias.size(0) == K, "Provided bias.size(0) must match weights.size(0) (output channels K)");
            bias_ptr_for_kernel = bias.data_ptr<float>();
        }
        // If bias.defined() is true but bias.numel() == 0 (i.e., torch.empty(0) was passed),
        // then bias_ptr_for_kernel remains nullptr, which is correct for "no bias".
    }
    // If bias was not defined at all (e.g., from an optional<Tensor> that's nullopt),
    // bias_ptr_for_kernel also remains nullptr.

    cudaError_t err = conv3d_forward_cuda_launcher(
        input.data_ptr<float>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        bias_ptr_for_kernel, // Pass the carefully determined pointer
        N, C, D_in, H_in, W_in,
        K, D_f, H_f, W_f,
        D_out, H_out, W_out,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w
    );

    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv3d_forward_cpp, "Custom 3D Convolution forward (CUDA)");
}