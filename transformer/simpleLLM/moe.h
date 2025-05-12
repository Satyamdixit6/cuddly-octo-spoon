#ifndef MOE_H
#define MOE_H

#include <torch/extension.h>
#include <vector>

torch::Tensor mixture_of_experts(
    const torch::Tensor& input,
    const std::vector<torch::Tensor>& expert_weights,
    const std::vector<torch::Tensor>& expert_biases,
    int k
);

#endif // MOE_H