#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        TORCH_CHECK(false, "CUDA error"); \
    } \
} while(0)

__global__ void label_embed_kernel(const int* __restrict__ labels,
                                   const int* __restrict__ force_drop,
                                   const float* __restrict__ embedding_table,
                                   float* __restrict__ out,
                                   int N, int hidden_size, int num_classes) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (n < N && j < hidden_size) {
        int label = labels[n];
        if (force_drop[n] == 1) label = num_classes;
        out[n * hidden_size + j] = embedding_table[label * hidden_size + j];
    }
}

torch::Tensor label_embed_forward(torch::Tensor labels, torch::Tensor embedding_table, torch::Tensor force_drop) {
    CHECK_CUDA(labels); CHECK_CUDA(embedding_table); CHECK_CUDA(force_drop);
    CHECK_CONTIGUOUS(labels); CHECK_CONTIGUOUS(embedding_table); CHECK_CONTIGUOUS(force_drop);

    int N = labels.size(0);
    int hidden_size = embedding_table.size(1);
    int num_classes = embedding_table.size(0) - 1;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto out = torch::empty({N, hidden_size}, options);
    dim3 blockDim(32, 8);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (hidden_size + blockDim.y - 1) / blockDim.y);
    label_embed_kernel<<<gridDim, blockDim>>>(
        labels.data_ptr<int>(),
        force_drop.data_ptr<int>(),
        embedding_table.data_ptr<float>(),
        out.data_ptr<float>(),
        N, hidden_size, num_classes
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("label_embed_forward", &label_embed_forward, "Label Embedding forward (CUDA)");
}