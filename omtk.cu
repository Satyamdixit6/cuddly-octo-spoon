#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void transposeSharedKernel(const float* input, float* output, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = yIndex * width + xIndex;

    int xTranspose = blockIdx.y * TILE_DIM + threadIdx.x;
    int yTranspose = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = yTranspose * height + xTranspose;

    if (xIndex < width && yIndex < height) {
        tile[threadIdx.y][threadIdx.x] = input[index_in];
    }
    __syncthreads();

    if (xTranspose < height && yTranspose < width) {
         output[index_out] = tile[threadIdx.x][threadIdx.y];
    }
}