#ifndef ATTENTION_H
#define ATTENTION_H

#include <torch/extension.h>

torch::Tensor multi_head_self_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& mask,
    int num_heads
);

#endif // ATTENTION_H