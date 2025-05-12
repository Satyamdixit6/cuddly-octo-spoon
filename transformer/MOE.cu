#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel for top-k selection
__global__ void top_k_kernel(const float* scores, int* indices, float* values, int batch_size, int num_experts, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * num_experts) return;

    int b = idx / num_experts;  // Batch index

    // Perform top-k selection for each token
    for (int i = 0; i < k; ++i) {
        float max_val = -INFINITY;
        int max_idx = -1;
        for (int j = 0; j < num_experts; ++j) {
            if (scores[b * num_experts + j] > max_val) {
                max_val = scores[b * num_experts + j];
                max_idx = j;
            }
        }
        indices[b * k + i] = max_idx;
        values[b * k + i] = max_val;
        scores[b * num_experts + max_idx] = -INFINITY;  // Invalidate this expert
    }
}

// Function to implement Mixture of Experts
torch::Tensor mixture_of_experts(
    const torch::Tensor& input,
    const std::vector<torch::Tensor>& expert_weights,
    const std::vector<torch::Tensor>& expert_biases,
    int k
) {
    // Get dimensions
    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int embedding_dim = input.size(2);
    int num_experts = expert_weights.size();

    // Flatten input for gating computation
    auto input_flat = input.view({batch_size * seq_len, embedding_dim});

    // Gating function: Compute scores for each expert
    auto gating_scores = torch::matmul(input_flat, expert_weights[0].transpose(0, 1)) + expert_biases[0];
    gating_scores = torch::softmax(gating_scores, /*dim=*/1);

    // Top-k selection
    auto indices = torch::zeros({batch_size * seq_len, k}, torch::kInt32).cuda();
    auto values = torch::zeros({batch_size * seq_len, k}, torch::kFloat32).cuda();
    int total_elements = batch_size * seq_len * num_experts;

    // Launch top-k kernel
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    top_k_kernel<<<blocks, threads_per_block>>>(
        gating_scores.data_ptr<float>(),
        indices.data_ptr<int>(),
        values.data_ptr<float>(),
        batch_size * seq_len, num_experts, k
    );

    // Apply selected experts
    auto output_flat = torch::zeros_like(input_flat);
    for (int i = 0; i < k; ++i) {
        auto selected_indices = indices.index({torch::indexing::Slice(), i});
        auto selected_values = values.index({torch::indexing::Slice(), i});

        // Gather inputs for the selected experts
        auto selected_inputs = torch::index_select(input_flat, /*dim=*/0, selected_indices);

        // Apply expert transformation
        auto expert_output = torch::matmul(selected_inputs, expert_weights[i]) + expert_biases[i];

        // Scatter back to the output
        output_flat.index_add_(/*dim=*/0, selected_indices, expert_output * selected_values.unsqueeze(1));
    }

    // Reshape output back to original shape
    auto output = output_flat.view({batch_size, seq_len, embedding_dim});
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mixture_of_experts", &mixture_of_experts, "Mixture of Experts");
}