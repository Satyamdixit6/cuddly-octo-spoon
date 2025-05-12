
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrixMultiplication(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

int main() {
    const int M = 4;
    const int N = 2;
    const int K = 4;

    float *h_a = (float*)malloc(M * N * sizeof(float));
    float *h_b = (float*)malloc(N * K * sizeof(float));
    float *h_c = (float*)malloc(M * K * sizeof(float));

    for (int i = 0; i < M * N; ++i) h_a[i] = static_cast<float>(i % 5);
    for (int i = 0; i < N * K; ++i) h_b[i] = static_cast<float>(i % 3);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, M * N * sizeof(float));
    cudaMalloc(&d_b, N * K * sizeof(float));
    cudaMalloc(&d_c, M * K * sizeof(float));

    cudaMemcpy(d_a, h_a, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    matrixMultiplication<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);

    cudaMemcpy(h_c, d_c, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}