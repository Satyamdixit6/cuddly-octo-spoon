#include <cuda_runtime.h>

__constant__ float constant_filter[25];

__global__ void applyConstantFilterKernel(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        int filter_radius = 2;

        for (int fy = -filter_radius; fy <= filter_radius; ++fy) {
            for (int fx = -filter_radius; fx <= filter_radius; ++fx) {
                int ix = x + fx;
                int iy = y + fy;
                int filter_idx = (fy + filter_radius) * 5 + (fx + filter_radius);

                ix = max(0, min(width - 1, ix));
                iy = max(0, min(height - 1, iy));

                sum += input[iy * width + ix] * constant_filter[filter_idx];
            }
        }
        output[y * width + x] = sum;
    }
}