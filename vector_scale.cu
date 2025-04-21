#include <cuda_runtime.h>

__global__ void vectorScaleKernel(float* vec, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        vec[idx] *= scalar;
    }
}