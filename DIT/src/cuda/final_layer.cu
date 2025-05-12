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

__global__ void silu_kernel(const float* __restrict__ inp,
                            float* __restrict__ out,
                            int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = inp[idx];
        out[idx] = silu(val);
    }
}

__global__ void add_bias_kernel(float* __restrict__ data,
                                const float* __restrict__ bias,
                                int total, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        int c = idx % dim;
        data[idx] += bias[c];
    }
}

__global__ void ln_kernel(
    const float* __restrict__ inp,
    float* __restrict__ out,
    int B, int T, int C, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T) return;
    int b_t = idx;
    int b_ = b_t / T;
    int t_ = b_t % T;
    const float* row_in = inp + ((long long)b_ * T + t_) * C;
    float* row_out = out + ((long long)b_ * T + t_) * C;

    float mean = 0.f;
    for (int i = 0; i < C; i++) {
        mean += row_in[i];
    }
    mean /= C;

    float var = 0.f;
    for (int i = 0; i < C; i++) {
        float diff = row_in[i] - mean;
        var += diff * diff;
    }
    var /= C;
    float inv_std = rsqrtf(var + eps);

    for (int i = 0; i < C; i++) {
        row_out[i] = (row_in[i] - mean) * inv_std;
    }
}

__global__ void mod_kernel(
    const float* __restrict__ x_ln,
    const float* __restrict__ shift,
    const float* __restrict__ scale,
    float* __restrict__ x_mod,
    int B, int T, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * C) return;
    int c = idx % C;
    int b_t = idx / C;
    int b = b_t / T;
    float val = x_ln[idx];
    val = val * scale[b * C + c] + shift[b * C + c];
    x_mod[idx] = val;
}

torch::Tensor final_layer_forward(
    torch::Tensor x, torch::Tensor c,
    torch::Tensor w_mod, torch::Tensor b_mod,
    torch::Tensor w_linear, torch::Tensor b_linear,
    float eps
) {
    CHECK_CUDA(x); CHECK_CUDA(c); CHECK_CUDA(w_mod); CHECK_CUDA(b_mod);
    CHECK_CUDA(w_linear); CHECK_CUDA(b_linear);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(c); CHECK_CONTIGUOUS(w_mod);
    CHECK_CONTIGUOUS(b_mod); CHECK_CONTIGUOUS(w_linear); CHECK_CONTIGUOUS(b_linear);

    int B = x.size(0);
    int T = x.size(1);
    int C_in = x.size(2);
    int out_dim = w_linear.size(1);

    auto opts = x.options();
    auto c_silu = torch::empty_like(c);
    {
        int total = B * C_in;
        dim3 block(256);
        dim3 grid((total + block.x - 1) / block.x);
        silu_kernel<<<grid, block>>>(c.data_ptr<float>(), c_silu.data_ptr<float>(), total);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    auto mod_out = torch::empty({B, 2 * C_in}, opts);
    {
        float alpha = 1.f;
        float beta = 0.f;
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            2 * C_in, B, C_in,
            &alpha,
            w_mod.data_ptr<float>(), 2 * C_in,
            c_silu.data_ptr<float>(), C_in,
            &beta,
            mod_out.data_ptr<float>(), 2 * C_in
        );
        int total = B * (2 * C_in);
        dim3 block(256);
        dim3 grid((total + block.x - 1) / block.x);
        add_bias_kernel<<<grid, block>>>(
            mod_out.data_ptr<float>(),
            b_mod.data_ptr<float>(),
            total,
            2 * C_in
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    cublasDestroy(handle);

    auto chunked = mod_out.chunk(2, 1);
    auto shift = chunked[0];
    auto scale = chunked[1];

    auto x_ln = torch::empty_like(x);
    {
        const int total = B * T;
        dim3 block(256);
        dim3 grid((total + block.x - 1) / block.x);
        ln_kernel<<<grid, block>>>(x.data_ptr<float>(), x_ln.data_ptr<float>(), B, T, C_in, eps);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    {
        int total2 = B * T * C_in;
        dim3 block(256);
        dim3 grid((total2 + block.x - 1) / block.x);
        mod_kernel<<<grid, block>>>(
            x_ln.data_ptr<float>(),
            shift.data_ptr<float>(),
            scale.data_ptr<float>(),
            x_ln.data_ptr<float>(),
            B, T, C_in
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    auto x_flat = x_ln.reshape({B * T, C_in});
    auto out = torch::empty({B * T, out_dim}, opts);

    {
        cublasHandle_t handle2;
        cublasCreate(&handle2);
        float alpha = 1.f;
        float beta = 0.f;
        cublasSgemm(
            handle2,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            out_dim, B * T, C_in,
            &alpha,
            w_linear.data_ptr<float>(), out_dim,
            x_flat.data_ptr<float>(), C_in,
            &beta,
            out.data_ptr<float>(), out_dim
        );
        int total = (B * T) * out_dim;
        dim3 block(256);
        dim3 grid((total + block.x - 1) / block.x);
        add_bias_kernel<<<grid, block>>>(
            out.data_ptr<float>(),
            b_linear.data_ptr<float>(),
            total,
            out_dim
        );
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        cublasDestroy(handle2);
    }

    return out.reshape({B, T, out_dim});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("final_layer_forward", &final_layer_forward, "Final Layer forward (CUDA)");
}