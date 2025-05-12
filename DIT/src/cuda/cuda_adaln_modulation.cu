#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
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

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void silu_kernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        out[idx] = val / (1.f + expf(-val));
    }
}

__global__ void add_bias_kernel(float* data, const float* bias, int total, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int c = idx % dim;
        data[idx] += bias[c];
    }
}

__global__ void ada_layernorm_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ shift,
    const float* __restrict__ scale,
    int B, int T, int C, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) return;

    int b_t = idx;
    const float* x_ptr = x + (long long)b_t * C;
    float mean = 0.f;
    float var = 0.f;

    for (int i = 0; i < C; i++) {
        mean += x_ptr[i];
    }
    mean /= C;

    for (int i = 0; i < C; i++) {
        float diff = x_ptr[i] - mean;
        var += diff * diff;
    }
    var /= C;

    float inv_std = rsqrtf(var + eps);
    int b_idx = b_t / T;
    const float* shift_ptr = shift + (long long)b_idx * C;
    const float* scale_ptr = scale + (long long)b_idx * C;
    float* y_ptr = y + (long long)b_t * C;

    for (int i = 0; i < C; i++) {
        float val = (x_ptr[i] - mean) * inv_std;
        val = val * scale_ptr[i] + shift_ptr[i];
        y_ptr[i] = val;
    }
}

torch::Tensor silu_linear_forward(torch::Tensor c, torch::Tensor w, torch::Tensor b) {
    CHECK_CUDA(c); CHECK_CUDA(w); CHECK_CUDA(b);
    CHECK_CONTIGUOUS(c); CHECK_CONTIGUOUS(w); CHECK_CONTIGUOUS(b);
    TORCH_CHECK(c.dtype() == torch::kFloat32, "float32 only");

    int B = c.size(0);
    int d = c.size(1);
    int out_dim = w.size(1);

    auto c_silu = torch::empty_like(c);
    {
        int threads = 256;
        int blocks = (B * d + threads - 1) / threads;
        silu_kernel<<<blocks, threads>>>(c.data_ptr<float>(), c_silu.data_ptr<float>(), B * d);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    auto out = torch::empty({B, out_dim}, c.options());
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.f;
    float beta = 0.f;
    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        out_dim, B, d,
        &alpha,
        w.data_ptr<float>(), out_dim,
        c_silu.data_ptr<float>(), d,
        &beta,
        out.data_ptr<float>(), out_dim
    );

    {
        int total = B * out_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        add_bias_kernel<<<blocks, threads>>>(out.data_ptr<float>(), b.data_ptr<float>(), total, out_dim);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    cublasDestroy(handle);
    return out;
}

torch::Tensor adaln_forward(torch::Tensor x, torch::Tensor shift, torch::Tensor scale, float eps) {
    CHECK_CUDA(x); CHECK_CUDA(shift); CHECK_CUDA(scale);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(shift); CHECK_CONTIGUOUS(scale);

    int B = x.size(0);
    int T = x.size(1);
    int C = x.size(2);

    auto y = torch::empty_like(x);
    int threads = 256;
    int blocks = (B * T + threads - 1) / threads;

    ada_layernorm_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        shift.data_ptr<float>(),
        scale.data_ptr<float>(),
        B, T, C, eps
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_linear_forward", &silu_linear_forward, "SiLU + Linear");
    m.def("adaln_forward", &adaln_forward, "Adaptive LN forward (CUDA)");
}