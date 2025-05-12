#include <torch/extension.h>
#include "attention.h"
#include "moe.h"

// Function for the full Transformer model
torch::Tensor transformer_model(
    const torch::Tensor& embeddings,
    const std::vector<int>& num_heads_list,
    const std::vector<int>& num_experts_list,
    int expert_hidden_dim
) {
    auto x = embeddings;

    for (size_t i = 0; i < num_heads_list.size(); ++i) {
        int num_heads = num_heads_list[i];
        int num_experts = num_experts_list[i];

        // Step 1: Multi-head self-attention
        auto attention_output = multi_head_self_attention(x, x, x, /*mask=*/{}, num_heads);

        // Step 2: Mixture of Experts (MoE)
        std::vector<torch::Tensor> expert_weights(num_experts);
        std::vector<torch::Tensor> expert_biases(num_experts);
        for (int j = 0; j < num_experts; ++j) {
            expert_weights[j] = torch::randn({x.size(2), expert_hidden_dim}).cuda();
            expert_biases[j] = torch::randn({expert_hidden_dim}).cuda();
        }
        auto moe_output = mixture_of_experts(attention_output, expert_weights, expert_biases, /*k=*/2);

        // Add residual connection
        x = x + moe_output;
    }

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multi_head_self_attention", &multi_head_self_attention, "Multi-head self-attention");
    m.def("mixture_of_experts", &mixture_of_experts, "Mixture of Experts");
    m.def("transformer_model", &transformer_model, "Full Transformer model");
}