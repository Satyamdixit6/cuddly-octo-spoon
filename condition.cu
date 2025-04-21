#include <cuda_runtime.h>

__global__ void conditionalKernel(const float* input, float* output, int n, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (input[idx] > threshold) {
            output[idx] = input[idx];
        } else {
            output[idx] = 0.0f;
        }
    }
}