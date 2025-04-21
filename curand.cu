#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void generateRandomKernel(float* output, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        curandState_t state;
        curand_init(seed, idx, 0, &state);
        output[idx] = curand_uniform(&state);
    }
}