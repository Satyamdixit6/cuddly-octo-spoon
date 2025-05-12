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

#define TILE_DIM 16

__global__ void timestep_embed_kernel(const float* __restrict__ t,
                                     const float* __restrict__ freqs,
                                     float* __restrict__ embed,
                                     int N, int dim, int half) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    if (n < N && d < dim) {
        __shared__ float freqs_shared[256];
        for (int i = threadIdx.y; i < half; i += blockDim.y) {
            freqs_shared[i] = freqs[i];
        }
        __syncthreads();

        float t_val = t[n];
        int index = n * dim + d;
        if (d < half) {
            embed[index] = cosf(t_val * freqs_shared[d]);
        } else {
            embed[index] = sinf(t_val * freqs_shared[d - half]);
        }
    }
}

__global__ void linear_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             const float* __restrict__ bias,
                             float* __restrict__ out,
                             int N, int K, int M) {
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        int A_col = t * TILE_DIM + threadIdx.x;
        if (row < N && A_col < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + A_col];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        int B_row = t * TILE_DIM + threadIdx.y;
        if (col < M && B_row < K)
            sB[threadIdx.y][threadIdx.x] = B[col * K + B_row];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_DIM; i++) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < M) {
        out[row * M + col] = sum + bias[col];
    }
}

__global__ void linear_silu_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  const float* __restrict__ bias,
                                  float* __restrict__ out,
                                  int N, int K, int M) {
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;
    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        int A_col = t * TILE_DIM + threadIdx.x;
        if (row < N && A_col < K)
            sA[threadIdx.y][threadIdx.x] = A[row * K + A_col];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        int B_row = t * TILE_DIM + threadIdx.y;
        if (col < M && B_row < K)
            sB[threadIdx.y][threadIdx.x] = B[col * K + B_row];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_DIM; i++) {
            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < M) {
        float x = sum + bias[col];
        float sig = 1.f / (1.f + expf(-x));
        out[row * M + col] = x * sig;
    }
}

torch::Tensor timestep_embedding_forward(torch::Tensor t, torch::Tensor freqs, int dim) {
    CHECK_CUDA(t); CHECK_CUDA(freqs);
    CHECK_CONTIGUOUS(t); CHECK_CONTIGUOUS(freqs);

    int N = t.size(0);
    int half = dim / 2;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto embed = torch::empty({N, dim}, options);
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (dim + blockDim.y - 1) / blockDim.y);
    timestep_embed_kernel<<<gridDim, blockDim>>>(
        t.data_ptr<float>(), freqs.data_ptr<float>(), embed.data_ptr<float>(), N, dim, half
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return embed;
}

torch::Tensor linear_forward(torch::Tensor A, torch::Tensor weight, torch::Tensor bias) {
    CHECK_CUDA(A); CHECK_CUDA(weight); CHECK_CUDA(bias);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(weight); CHECK_CONTIGUOUS(bias);

    int N = A.size(0);
    int K = A.size(1);
    int M = weight.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto out = torch::empty({N, M}, options);
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    linear_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), N, K, M
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return out;
}

torch::Tensor linear_silu_forward(torch::Tensor A, torch::Tensor weight, torch::Tensor bias) {
    CHECK_CUDA(A); CHECK_CUDA(weight); CHECK_CUDA(bias);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(weight); CHECK_CONTIGUOUS(bias);

    int N = A.size(0);
    int K = A.size(1);
    int M = weight.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto out = torch::empty({N, M}, options);
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    linear_silu_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), N, K, M
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return out;
}

torch::Tensor timestep_embed_mlp_forward(torch::Tensor t_freq,
                                        torch::Tensor weight1, torch::Tensor bias1,
                                        torch::Tensor weight2, torch::Tensor bias2) {
    auto hidden = linear_silu_forward(t_freq, weight1, bias1);
    auto out = linear_forward(hidden, weight2, bias2);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("timestep_embedding_forward", &timestep_embedding_forward, "Timestep embedding forward (CUDA)");
    m.def("linear_forward", &linear_forward, "Linear forward (CUDA)");
    m.def("linear_silu_forward", &linear_silu_forward, "Linear + SiLU forward (CUDA)");
    m.def("timestep_embed_mlp_forward", &timestep_embed_mlp_forward, "Full MLP forward (CUDA)");
}