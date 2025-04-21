#include <cuda_runtime.h>

__global__ void reduceSumNaiveKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        *output = 0.0f;
    }
    __syncthreads();

    if (idx < n) {
        atomicAdd(output, input[idx]);
    }
}