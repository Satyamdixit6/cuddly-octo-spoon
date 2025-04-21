#include <torch/extension.h>

#include <cuda.h>
include <cuda_runtime.h>
#include <cmath>

__global__ void softmax_kernel(const flaot)