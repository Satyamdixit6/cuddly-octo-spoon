#include <cuda_runtime.h>
#include <math.h>

__global__ void elementwiseSquareKernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * input[idx];
    }
}