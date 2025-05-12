#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        TORCH_CHECK(false, "CUDA error"); \
    } \
} while(0)

using namespace nvcuda::wmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK 4

__device__ __forceinline__ half gelu(half x) {
    const half ALPHA = __float2half(0.044715f);
    const half BETA = __float2half(0.79788456f);
    half x3 = x * x * x;
    half inner = BETA * (x + ALPHA * x3);
    return x * __float2half(0.5f) * (half(1.0f) + __htan(inner));
}

__global__ void __launch_bounds__(BLOCK_SIZE) fused_mlp_persistent(
    const half* __restrict__ input,
    const half* __restrict__ weight1,
    const half* __restrict__ bias1,
    const half* __restrict__ weight2,
    const half* __restrict__ bias2,
    half* __restrict__ output,
    int batch_size, int seq_len, int in_dim, int hidden_dim, int out_dim) {
    __shared__ half shared_input[WMMA_M][WMMA_K];
    __shared__ half shared_weight[WMMA_K][WMMA_N];
    __shared__ half shared_output[WMMA_M][WMMA_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    for (int batch_idx = blockIdx.x; batch_idx < batch_size * seq_len; batch_idx += gridDim.x) {
        const int batch = batch_idx / seq_len;
        const int seq = batch_idx % seq_len;

        for (int hidden_offset = 0; hidden_offset < hidden_dim; hidden_offset += WMMA_N) {
            wmma::fill_fragment(acc_frag, __float2half(0.0f));

            for (int k = 0; k < in_dim; k += WMMA_K) {
                if (threadIdx.x < WMMA_M) {
                    #pragma unroll
                    for (int i = 0; i < WMMA_K; i += 4) {
                        reinterpret_cast<float2*>(&shared_input[threadIdx.x][i])[0] =
                            reinterpret_cast<const float2*>(&input[batch_idx * in_dim + k + i])[0];
                    }
                }
                if (threadIdx.x < WMMA_K) {
                    #pragma unroll
                    for (int i = 0; i < WMMA_N; i += 4) {
                        reinterpret_cast<float2*>(&shared_weight[threadIdx.x][i])[0] =
                            reinterpret_cast<const float2*>(&weight1[(hidden_offset + i) * in_dim + k + threadIdx.x])[0];
                    }
                }
                __syncthreads();

                wmma::load_matrix_sync(a_frag, (half*)shared_input, WMMA_K);
                wmma::load_matrix_sync(b_frag, (half*)shared_weight, WMMA_N);
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
                __syncthreads();
            }

            for (int i = 0; i < acc_frag.num_elements; i++) {
                acc_frag.x[i] = gelu(acc_frag.x[i] + bias1[hidden_offset + i]);
            }

            wmma::store_matrix_sync(shared_output, acc_frag, WMMA_N, wmma::mem_row_major);
            __syncthreads();

            if (threadIdx.x < WMMA_M) {
                #pragma unroll
                for (int i = 0; i < WMMA_N; i += 4) {
                    reinterpret_cast<float2*>(&output[batch_idx * hidden_dim + hidden_offset + i])[0] =
                        reinterpret_cast<float2*>(&shared_output[threadIdx.x][i])[0];
                }
            }
            __syncthreads();
        }
    }
}

torch::Tensor mlp_forward(torch::Tensor x, torch::Tensor w1, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2) {
    CHECK_CUDA(x); CHECK_CUDA(w1); CHECK_CUDA(b1); CHECK_CUDA(w2); CHECK_CUDA(b2);
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(w1); CHECK_CONTIGUOUS(b1); CHECK_CONTIGUOUS(w2); CHECK_CONTIGUOUS(b2);

    const int batch_size = x.size(0);
    const int seq_len = x.size(1);
    const int in_dim = x.size(2);
    const int hidden_dim = w1.size(1);
    const int out_dim = w2.size(1);

    auto x_half = x.to(torch::kHalf);
    auto w1_half = w1.to(torch::kHalf);
    auto b1_half = b1.to(torch::kHalf);
    auto w2_half = w2.to(torch::kHalf);
    auto b2_half = b2.to(torch::kHalf);

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
    auto output = torch::empty({batch_size, seq_len, out_dim}, options);

    dim3 grid(std::min(batch_size * seq_len, 256));
    dim3 block(BLOCK_SIZE);

    fused_mlp_persistent<<<grid, block>>>(
        x_half.data_ptr<half>(),
        w1_half.data_ptr<half>(),
        b1_half.data_ptr<half>(),
        w2_half.data_ptr<half>(),
        b2_half.data_ptr<half>(),
        output.data_ptr<half>(),
        batch_size, seq_len, in_dim, hidden_dim, out_dim
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return output.to(torch::kFloat);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mlp_forward", &mlp_forward, "Optimized MLP forward (CUDA)");
}