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

#define THREADS 256

__global__ void attention_forward_kernel(const float* __restrict__ Q,
                                        const float* __restrict__ K,
                                        const float* __restrict__ V,
                                        float* __restrict__ out,
                                        int B, int T, int C) {
    int b = blockIdx.x;
    int i = blockIdx.y;
    int tid = threadIdx.x;
    extern __shared__ float sdata[];
    float* scores = sdata;

    for (int j = tid; j < T; j += blockDim.x) {
        float dot = 0.0f;
        int q_offset = b * T * C + i * C;
        int k_offset = b * T * C + j * C;
        for (int k = 0; k < C; k++) {
            dot += Q[q_offset + k] * K[k_offset + k];
        }
        scores[j] = dot / sqrtf((float)C);
    }
    __syncthreads();

    float max_val = -1e20f;
    for (int j = tid; j < T; j += blockDim.x) {
        if (scores[j] > max_val) max_val = scores[j];
    }
    __shared__ float smax[THREADS];
    smax[tid] = max_val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            if (smax[tid + stride] > smax[tid]) smax[tid] = smax[tid + stride];
        }
        __syncthreads();
    }
    max_val = smax[0];

    float sum = 0.0f;
    for (int j = tid; j < T; j += blockDim.x) {
        float exp_val = expf(scores[j] - max_val);
        scores[j] = exp_val;
        sum += exp_val;
    }
    __shared__ float ssum[THREADS];
    ssum[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            ssum[tid] += ssum[tid + stride];
        }
        __syncthreads();
    }
    sum = ssum[0];

    for (int j = tid; j < T; j += blockDim.x) {
        scores[j] /= sum;
    }
    __syncthreads();

    if (tid == 0) {
        for (int k = 0; k < C; k++) {
            float acc = 0.0f;
            for (int j = 0; j < T; j++) {
                int v_offset = b * T * C + j * C;
                acc += scores[j] * V[v_offset + k];
            }
            out[b * T * C + i * C + k] = acc;
        }
    }
}

torch::Tensor attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    CHECK_CUDA(Q); CHECK_CUDA(K); CHECK_CUDA(V);
    CHECK_CONTIGUOUS(Q); CHECK_CONTIGUOUS(K); CHECK_CONTIGUOUS(V);
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Only float32 supported");

    int B = Q.size(0);
    int T = Q.size(1);
    int C = Q.size(2);

    auto out = torch::empty({B, T, C}, Q.options());
    dim3 grid(B, T);
    dim3 block(THREADS);
    size_t shared_memory_size = T * sizeof(float);
    attention_forward_kernel<<<grid, block, shared_memory_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        out.data_ptr<float>(),
        B, T, C
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attention_forward", &attention_forward, "Attention Forward (CUDA)");
}