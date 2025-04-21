#include <cuda_runtime.h>

__global__ void histogramAtomicKernel(const unsigned int* input, unsigned int* bins, int n, int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int value = input[idx];
        if (value < num_bins) {
            atomicAdd(&bins[value], 1);
        }
    }
}