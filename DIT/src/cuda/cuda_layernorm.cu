#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        TORCH_CHECK(false, "CUDA error"); \
    } \
} while(0)

#define BLOCK_SIZE 256
#define WARP_SIZE 32

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    __shared__ float blockSum;
    if (threadIdx.x == 0) blockSum = val;
    __syncthreads();
    val = blockSum;
    return val;
}

__global__ void layernorm_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    const int N, const int C
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (bid >= N) return;

    const float* input_row = input + bid * C;
    float* output_row = output + bid * C;

    float sum = 0.0f;
    for (int i = tid; i < C; i += BLOCK_SIZE) {
        sum += input_row[i];
    }
    sum = blockReduceSum(sum);
    float mu = sum / C;

    __shared__ float mu_shared;
    if (tid == 0) {
        mean[bid] = mu;
        mu_shared = mu;
    }
    __syncthreads();
    mu = mu_shared;

    float var_sum = 0.0f;
    for (int i = tid; i < C; i += BLOCK_SIZE) {
        float diff = input_row[i] - mu;
        var_sum += diff * diff;
    }
    var_sum = blockReduceSum(var_sum);
    float sigma = sqrtf(var_sum / C + 1e-5f);
    float inv_sigma = 1.0f / sigma;

    __shared__ float inv_sigma_shared;
    if (tid == 0) {
        rstd[bid] = inv_sigma;
        inv_sigma_shared = inv_sigma;
    }
    __syncthreads();
    inv_sigma = inv_sigma_shared;

    for (int i = tid; i < C; i += BLOCK_SIZE) {
        output_row[i] = (input_row[i] - mu) * inv_sigma;
    }
}

torch::Tensor layernorm_forward(torch::Tensor input) {
    CHECK_CUDA(input);
    CHECK_CONTIGUOUS(input);
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (batch_size, hidden_size)");
    TORCH_CHECK(input.scalar_type() == torch::ScalarType::Float, "Input must be float32");

    const int N = input.size(0);
    const int C = input.size(1);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto output = torch::empty_like(input);
    auto mean = torch::empty({N}, options);
    auto rstd = torch::empty({N}, options);

    const dim3 blocks(N);
    const dim3 threads(BLOCK_SIZE);
    layernorm_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        mean.data_ptr<float>(),
        rstd.data_ptr<float>(),
        N, C
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layernorm_forward", &layernorm_forward, "LayerNorm forward (CUDA)");
}