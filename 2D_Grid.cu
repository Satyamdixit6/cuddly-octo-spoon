#include <cuda_runtime.h>

__global__ void kernel2D(float* data, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        data[idx] = (float)(x + y * width);
    }
}