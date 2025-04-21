#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matMulTiledKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float Cvalue = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        if (row < M && (t * TILE_WIDTH + tx) < K) {
            As[ty][tx] = A[row * K + (t * TILE_WIDTH + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        if ((t * TILE_WIDTH + ty) < K && col < N) {
            Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
    }
}