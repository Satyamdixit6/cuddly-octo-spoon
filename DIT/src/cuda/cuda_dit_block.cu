#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        TORCH_CHECK(false, "CUDA error"); \
    } \
} while(0)

// Declare external functions (implemented in other .cu files)
extern torch::Tensor silu_linear_forward(torch::Tensor c, torch::Tensor w, torch::Tensor b);
extern torch::Tensor adaln_forward(torch::Tensor x, torch::Tensor shift, torch::Tensor scale, float eps);
extern torch::Tensor attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
extern torch::Tensor mlp_forward(torch::Tensor x, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2);
extern torch::Tensor linear_forward(torch::Tensor A, torch::Tensor weight, torch::Tensor bias);

torch::Tensor dit_block_forward(
    torch::Tensor x, torch::Tensor c,
    torch::Tensor w_mod, torch::Tensor b_mod,
    torch::Tensor wQ, torch::Tensor bQ,
    torch::Tensor wK, torch::Tensor bK,
    torch::Tensor wV, torch::Tensor bV,
    torch::Tensor wO, torch::Tensor bO,
    torch::Tensor w1, torch::Tensor b1,
    torch::Tensor w2, torch::Tensor b2,
    float eps
) {
    CHECK_CUDA(x); CHECK_CUDA(c); CHECK_CUDA(w_mod); CHECK_CUDA(b_mod);
    CHECK_CUDA(wQ); CHECK_CUDA(bQ); CHECK_CUDA(wK); CHECK_CUDA(bK);
    CHECK_CUDA(wV); CHECK_CUDA(bV); CHECK_CUDA(wO); CHECK_CUDA(bO);
    CHECK_CUDA(w1); CHECK_CUDA(b1); CHECK_CUDA(w2); CHECK_CUDA(b2);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(c); CHECK_CONTIGUOUS(w_mod);
    CHECK_CONTIGUOUS(b_mod); CHECK_CONTIGUOUS(wQ); CHECK_CONTIGUOUS(bQ);
    CHECK_CONTIGUOUS(wK); CHECK_CONTIGUOUS(bK); CHECK_CONTIGUOUS(wV);
    CHECK_CONTIGUOUS(bV); CHECK_CONTIGUOUS(wO); CHECK_CONTIGUOUS(bO);
    CHECK_CONTIGUOUS(w1); CHECK_CONTIGUOUS(b1); CHECK_CONTIGUOUS(w2); CHECK_CONTIGUOUS(b2);

    auto sixC = silu_linear_forward(c, w_mod, b_mod);
    auto parts = sixC.chunk(6, 1);
    auto shift_msa = parts[0];
    auto scale_msa = parts[1];
    auto gate_msa = parts[2];
    auto shift_mlp = parts[3];
    auto scale_mlp = parts[4];
    auto gate_mlp = parts[5];

    auto x_ln = adaln_forward(x, shift_msa, scale_msa, eps);
    int B = x.size(0);
    int T = x.size(1);
    int C_ = x.size(2);

    auto x_ln_flat = x_ln.reshape({B * T, C_});
    auto Q_ = linear_forward(x_ln_flat, wQ, bQ).reshape({B, T, C_});
    auto K_ = linear_forward(x_ln_flat, wK, bK).reshape({B, T, C_});
    auto V_ = linear_forward(x_ln_flat, wV, bV).reshape({B, T, C_});

    auto attn = attention_forward(Q_, K_, V_);
    auto attn_flat = attn.reshape({B * T, C_});
    auto out_attn = linear_forward(attn_flat, wO, bO).reshape({B, T, C_});

    x = x + gate_msa.unsqueeze(1) * out_attn;
    auto x_ln2 = adaln_forward(x, shift_mlp, scale_mlp, eps);
    auto x_mlp = mlp_forward(x_ln2, w1, b1, w2, b2);
    x = x + gate_mlp.unsqueeze(1) * x_mlp;

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dit_block_forward", &dit_block_forward, "DiT Block forward (CUDA)");
}