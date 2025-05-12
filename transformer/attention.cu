#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel for softmax computation
__global__ void softmax_kernel(const float* input, float* output, int batch_size, int seq_len, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * seq_len * dim) return;

    int b = idx / (seq_len * dim);  // Batch index
    int s = (idx % (seq_len * dim)) / dim;  // Sequence index

    // Compute max value along the sequence dimension
    float max_val = -INFINITY;
    for (int i = 0; i < seq_len; ++i) {
        max_val = fmaxf(max_val, input[b * seq_len * dim + i * dim + (idx % dim)]);
    }

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int i = 0; i < seq_len; ++i) {
        float val = expf(input[b * seq_len * dim + i * dim + (idx % dim)] - max_val);
        sum_exp += val;
        output[b * seq_len * dim + i * dim + (idx % dim)] = val;
    }

    // Normalize
    for (int i = 0; i < seq_len; ++i) {
        output[b * seq_len * dim + i * dim + (idx % dim)] /= sum_exp;
    }
}

// Function to compute multi-head self-attention
torch::Tensor multi_head_self_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& mask,
    int num_heads
) {
    // Get dimensions
    int batch_size = query.size(0);
    int seq_len = query.size(1);
    int embedding_dim = query.size(2);

    // Split embedding_dim into num_heads
    int head_dim = embedding_dim / num_heads;

    // Reshape inputs for multi-head computation
    auto query_reshaped = query.view({batch_size, seq_len, num_heads, head_dim}).permute({0, 2, 1, 3});
    auto key_reshaped = key.view({batch_size, seq_len, num_heads, head_dim}).permute({0, 2, 1, 3});
    auto value_reshaped = value.view({batch_size, seq_len, num_heads, head_dim}).permute({0, 2, 1, 3});

    // Compute attention scores: Q @ K^T / sqrt(d_k)
    auto scores = torch::matmul(query_reshaped, key_reshaped.transpose(-2, -1)) / sqrt((float)head_dim);

    // Apply mask (if provided)
    if (mask.defined()) {
        scores += mask;
    }

    // Apply softmax
    auto scores_flat = scores.contiguous().view({-1});
    auto probs_flat = torch::empty_like(scores_flat).cuda();
    int total_elements = scores.numel();

    // Launch softmax kernel
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    softmax_kernel<<<blocks, threads_per_block>>>(scores_flat.data_ptr<float>(), probs_flat.data_ptr<float>(), batch_size, seq_len, head_dim);

    auto probs = probs_flat.view_as(scores);

    // Compute weighted sum: softmax(Q @ K^T / sqrt(d_k)) @ V
    auto output = torch::matmul(probs, value_reshaped);

    // Reshape back to original shape
    output = output.permute({0, 2, 1, 3}).contiguous().view({batch_size, seq_len, embedding_dim});

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multi_head_self_attention", &multi_head_self_attention, "Multi-head self-attention");
}