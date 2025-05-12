// transformer_block.cu
// This file uses PyTorch's C++ API (ATen) to implement a Transformer block.
// When input tensors are on a CUDA device, PyTorch dispatches operations
// to its CUDA backend (cuBLAS, cuDNN, custom kernels).

#include <torch/extension.h>
#include <vector>
#include <cmath> // For std::sqrt

// Helper function for Layer Normalization Forward (simplified)
// PyTorch's built-in layer_norm is generally preferred and optimized.
// This is illustrative.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> layer_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps = 1e-5)
{
    // Ensure input is contiguous for reliable mean/variance calculation
    x = x.contiguous();
    // Calculate mean and variance along the last dimension (feature dimension)
    auto mean = x.mean(-1, /*keepdim=*/true);
    // Use unbiased=false for consistency with common implementations
    auto variance = x.var(-1, /*unbiased=*/false, /*keepdim=*/true);
    auto inv_std = (variance + eps).rsqrt(); // Inverse standard deviation

    // Normalize the input tensor
    auto x_normalized = (x - mean) * inv_std;

    // Scale and shift using learnable parameters (weight and bias)
    auto out = x_normalized * weight + bias;

    // Return output, mean, and inverse standard deviation for backward pass
    return std::make_tuple(out, mean, inv_std);
}

// Helper function for Layer Normalization Backward (simplified)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> layer_norm_backward(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor mean,
    torch::Tensor inv_std,
    int64_t hidden_dim)
{
    // Ensure input is contiguous
    x = x.contiguous();
    // Calculate gradients w.r.t. weight and bias
    auto x_normalized = (x - mean) * inv_std;
    auto grad_weight = (grad_out * x_normalized).sum(0); // Sum over batch & sequence length
    auto grad_bias = grad_out.sum(0); // Sum over batch & sequence length

    // Calculate gradient w.r.t. the normalized input
    auto grad_x_normalized = grad_out * weight;

    // Calculate gradient w.r.t. variance (intermediate step)
    auto grad_inv_std = (grad_x_normalized * (x - mean)).sum(-1, true);
    auto grad_variance = grad_inv_std * (-0.5 * inv_std.pow(3));

    // Calculate gradient w.r.t. mean (intermediate step)
    auto grad_mean = (grad_x_normalized * (-inv_std)).sum(-1, true) +
                     grad_variance * (-2.0 * (x - mean) / hidden_dim).sum(-1, true);

    // Calculate gradient w.r.t. input x
    auto grad_x = grad_x_normalized * inv_std +
                  grad_variance * (2.0 * (x - mean) / hidden_dim) +
                  grad_mean / hidden_dim;

    return std::make_tuple(grad_x, grad_weight, grad_bias);
}


// Multi-Head Attention Implementation
class MultiHeadAttentionImpl : public torch::nn::Module {
public:
    MultiHeadAttentionImpl(int64_t embed_dim, int64_t num_heads, double dropout_p = 0.0)
        : embed_dim_(embed_dim),
          num_heads_(num_heads),
          head_dim_(embed_dim / num_heads),
          dropout_p_(dropout_p)
    {
        TORCH_CHECK(embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads");

        // Linear layers for Q, K, V projections and the final output projection
        qkv_proj_ = register_module("qkv_proj", torch::nn::Linear(embed_dim, embed_dim * 3));
        out_proj_ = register_module("out_proj", torch::nn::Linear(embed_dim, embed_dim));
        // Dropout layer
        attn_dropout_ = register_module("attn_dropout", torch::nn::Dropout(dropout_p_));
        resid_dropout_ = register_module("resid_dropout", torch::nn::Dropout(dropout_p_));

        // Initialize weights (example using Xavier uniform)
        torch::nn::init::xavier_uniform_(qkv_proj_->weight);
        torch::nn::init::xavier_uniform_(out_proj_->weight);
        if (qkv_proj_->bias.defined()) torch::nn::init::zeros_(qkv_proj_->bias);
        if (out_proj_->bias.defined()) torch::nn::init::zeros_(out_proj_->bias);
    }

    // Forward pass for Multi-Head Attention
    torch::Tensor forward(torch::Tensor x, c10::optional<torch::Tensor> attention_mask = c10::nullopt) {
        // x shape: [batch_size, seq_len, embed_dim]
        int64_t batch_size = x.size(0);
        int64_t seq_len = x.size(1);

        // 1. Project Q, K, V
        // Project x to Q, K, V combined: [batch_size, seq_len, embed_dim * 3]
        auto qkv = qkv_proj_->forward(x);
        // Split into Q, K, V: each [batch_size, seq_len, embed_dim]
        auto qkv_chunks = qkv.chunk(3, /*dim=*/-1);
        auto q = qkv_chunks[0];
        auto k = qkv_chunks[1];
        auto v = qkv_chunks[2];

        // 2. Reshape for Multi-Head Calculation
        // Reshape Q, K, V to [batch_size, num_heads, seq_len, head_dim]
        q = q.view({batch_size, seq_len, num_heads_, head_dim_}).transpose(1, 2);
        k = k.view({batch_size, seq_len, num_heads_, head_dim_}).transpose(1, 2);
        v = v.view({batch_size, seq_len, num_heads_, head_dim_}).transpose(1, 2);

        // 3. Scaled Dot-Product Attention
        // Calculate attention scores: Q @ K.T
        // q: [b, nh, sl, hd], k.transpose(-2, -1): [b, nh, hd, sl] -> scores: [b, nh, sl, sl]
        auto attn_scores = torch::matmul(q, k.transpose(-2, -1));
        // Scale the scores
        attn_scores /= std::sqrt(static_cast<double>(head_dim_));

        // Apply the attention mask (if provided)
        // Mask should be broadcastable to [b, nh, sl, sl]
        // Typically: [b, 1, sl, sl] or [b, 1, 1, sl] for causal masks
        if (attention_mask.has_value()) {
            // Add mask (usually large negative values for masked positions)
            attn_scores = attn_scores.masked_fill(attention_mask.value() == 0, -1e9); // Or -infinity
        }

        // Apply softmax to get attention probabilities
        auto attn_probs = torch::softmax(attn_scores, /*dim=*/-1);
        // Apply dropout to attention probabilities
        attn_probs = attn_dropout_->forward(attn_probs);

        // 4. Apply Attention to V
        // attn_probs: [b, nh, sl, sl], v: [b, nh, sl, hd] -> context: [b, nh, sl, hd]
        auto context = torch::matmul(attn_probs, v);

        // 5. Reshape and Project Output
        // Concatenate heads: [b, nh, sl, hd] -> [b, sl, nh, hd] -> [b, sl, embed_dim]
        context = context.transpose(1, 2).contiguous().view({batch_size, seq_len, embed_dim_});

        // Final linear projection
        auto output = out_proj_->forward(context);
        // Apply residual dropout
        output = resid_dropout_->forward(output);

        // output shape: [batch_size, seq_len, embed_dim]
        return output;
    }

private:
    int64_t embed_dim_;
    int64_t num_heads_;
    int64_t head_dim_;
    double dropout_p_;

    torch::nn::Linear qkv_proj_{nullptr}, out_proj_{nullptr};
    torch::nn::Dropout attn_dropout_{nullptr}, resid_dropout_{nullptr};
};
TORCH_MODULE(MultiHeadAttention); // Macro to create a module holder


// Feed-Forward Network Implementation
class FeedForwardImpl : public torch::nn::Module {
public:
    FeedForwardImpl(int64_t embed_dim, int64_t hidden_dim, double dropout_p = 0.0)
        : embed_dim_(embed_dim),
          hidden_dim_(hidden_dim),
          dropout_p_(dropout_p)
    {
        // Linear layers
        linear1_ = register_module("linear1", torch::nn::Linear(embed_dim_, hidden_dim_));
        linear2_ = register_module("linear2", torch::nn::Linear(hidden_dim_, embed_dim_));
        // Activation function (GELU is common in Transformers)
        activation_ = torch::nn::GELU();
        // Dropout layer
        dropout_ = register_module("dropout", torch::nn::Dropout(dropout_p_));

        // Initialize weights (example)
        torch::nn::init::xavier_uniform_(linear1_->weight);
        torch::nn::init::xavier_uniform_(linear2_->weight);
        if (linear1_->bias.defined()) torch::nn::init::zeros_(linear1_->bias);
        if (linear2_->bias.defined()) torch::nn::init::zeros_(linear2_->bias);
    }

    // Forward pass for Feed-Forward Network
    torch::Tensor forward(torch::Tensor x) {
        // x shape: [batch_size, seq_len, embed_dim]
        x = linear1_->forward(x);
        x = activation_->forward(x);
        // Apply dropout after activation but before the second linear layer
        x = dropout_->forward(x);
        x = linear2_->forward(x);
        // Apply dropout after the second linear layer (optional, sometimes done)
        // x = dropout_->forward(x); // Uncomment if dropout is desired here too
        // output shape: [batch_size, seq_len, embed_dim]
        return x;
    }

private:
    int64_t embed_dim_;
    int64_t hidden_dim_;
    double dropout_p_;

    torch::nn::Linear linear1_{nullptr}, linear2_{nullptr};
    torch::nn::GELU activation_{nullptr}; // Using GELU activation
    torch::nn::Dropout dropout_{nullptr};
};
TORCH_MODULE(FeedForward); // Macro to create a module holder


// --- Custom Autograd Function for LayerNorm (Illustrative) ---
// In practice, use torch::layer_norm which has optimized backward.
class LayerNormFunction : public torch::autograd::Function<LayerNormFunction> {
public:
    // Forward pass: Calculates LayerNorm and saves tensors for backward
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor input,
        torch::Tensor weight,
        torch::Tensor bias,
        double eps)
    {
        auto [output, mean, inv_std] = layer_norm_forward(input, weight, bias, eps);
        // Save necessary tensors for backward pass
        ctx->save_for_backward({input, weight, mean, inv_std});
        ctx->saved_data["hidden_dim"] = input.size(-1); // Save hidden dimension
        return output;
    }

    // Backward pass: Calculates gradients w.r.t. input, weight, and bias
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list grad_outputs)
    {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto mean = saved[2];
        auto inv_std = saved[3];
        int64_t hidden_dim = ctx->saved_data["hidden_dim"].toInt();

        auto grad_output = grad_outputs[0]; // Gradient from the next layer

        // Calculate gradients using the backward helper function
        auto [grad_input, grad_weight, grad_bias] = layer_norm_backward(
            grad_output, input, weight, mean, inv_std, hidden_dim);

        // Return gradients corresponding to inputs of the forward function
        // Order: input, weight, bias, eps (eps doesn't need grad)
        return {grad_input, grad_weight, grad_bias, torch::Tensor()};
    }
};

// Wrapper function to use the custom LayerNorm autograd function
torch::Tensor layer_norm_custom(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
    return LayerNormFunction::apply(input, weight, bias, eps);
}


// --- Transformer Block combining all components ---
class TransformerBlockImpl : public torch::nn::Module {
public:
    TransformerBlockImpl(int64_t embed_dim, int64_t num_heads, int64_t ffn_hidden_dim, double dropout_p = 0.1, double layer_norm_eps = 1e-5)
        : embed_dim_(embed_dim),
          num_heads_(num_heads),
          ffn_hidden_dim_(ffn_hidden_dim),
          dropout_p_(dropout_p),
          layer_norm_eps_(layer_norm_eps)
    {
        // Layer Normalization layers (using PyTorch's optimized version)
        // Pre-Normalization is common: Norm -> Attention -> Add -> Norm -> FFN -> Add
        ln1_ = register_module("ln1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim_}).eps(layer_norm_eps_)));
        ln2_ = register_module("ln2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim_}).eps(layer_norm_eps_)));

        // Multi-Head Self-Attention layer
        self_attn_ = register_module("self_attn", MultiHeadAttention(embed_dim_, num_heads_, dropout_p_));

        // Feed-Forward Network layer
        ffn_ = register_module("ffn", FeedForward(embed_dim_, ffn_hidden_dim_, dropout_p_));

        // Note: Positional embeddings are assumed to be added *before* this block.
    }

    // Forward pass for the Transformer Block
    torch::Tensor forward(torch::Tensor x, c10::optional<torch::Tensor> attention_mask = c10::nullopt) {
        // x shape: [batch_size, seq_len, embed_dim]

        // 1. Pre-Normalization & Multi-Head Self-Attention with Residual Connection
        torch::Tensor attn_output = self_attn_->forward(ln1_->forward(x), attention_mask);
        // Add residual connection
        x = x + attn_output; // First residual connection

        // 2. Pre-Normalization & Feed-Forward Network with Residual Connection
        torch::Tensor ffn_output = ffn_->forward(ln2_->forward(x));
        // Add residual connection
        x = x + ffn_output; // Second residual connection

        // output shape: [batch_size, seq_len, embed_dim]
        return x;
    }

    // --- Backward Pass ---
    // The backward pass is automatically handled by PyTorch's autograd system
    // when using standard torch::nn modules (LayerNorm, Linear, Dropout, etc.)
    // and built-in tensor operations (matmul, softmax, add, etc.).
    // Our custom LayerNormFunction also integrates with autograd.
    // No explicit backward function is needed here unless we define the entire
    // block as a single custom autograd::Function, which is more complex.

private:
    int64_t embed_dim_;
    int64_t num_heads_;
    int64_t ffn_hidden_dim_;
    double dropout_p_;
    double layer_norm_eps_;

    torch::nn::LayerNorm ln1_{nullptr}, ln2_{nullptr};
    MultiHeadAttention self_attn_{nullptr};
    FeedForward ffn_{nullptr};
};
TORCH_MODULE(TransformerBlock); // Macro to create a module holder


// --- Python Bindings using pybind11 ---
// This section makes the C++ modules usable from Python.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Bind the MultiHeadAttention module
    pybind11::class_<MultiHeadAttentionImpl, torch::nn::Module, std::shared_ptr<MultiHeadAttentionImpl>>(m, "MultiHeadAttention")
        .def(pybind11::init<int64_t, int64_t, double>(),
             pybind11::arg("embed_dim"), pybind11::arg("num_heads"), pybind11::arg("dropout_p") = 0.0)
        // Expose the forward method. Handle optional mask.
        .def("forward", [](MultiHeadAttentionImpl &self, torch::Tensor x, c10::optional<torch::Tensor> mask) {
            return self.forward(x, mask);
        }, pybind11::arg("x"), pybind11::arg("attention_mask") = pybind11::none(), "Forward pass for MultiHeadAttention");


    // Bind the FeedForward module
    pybind11::class_<FeedForwardImpl, torch::nn::Module, std::shared_ptr<FeedForwardImpl>>(m, "FeedForward")
        .def(pybind11::init<int64_t, int64_t, double>(),
             pybind11::arg("embed_dim"), pybind11::arg("hidden_dim"), pybind11::arg("dropout_p") = 0.0)
        .def("forward", &FeedForwardImpl::forward, pybind11::arg("x"), "Forward pass for FeedForward");

    // Bind the TransformerBlock module
    pybind11::class_<TransformerBlockImpl, torch::nn::Module, std::shared_ptr<TransformerBlockImpl>>(m, "TransformerBlock")
        .def(pybind11::init<int64_t, int64_t, int64_t, double, double>(),
             pybind11::arg("embed_dim"), pybind11::arg("num_heads"), pybind11::arg("ffn_hidden_dim"),
             pybind11::arg("dropout_p") = 0.1, pybind11::arg("layer_norm_eps") = 1e-5)
        // Expose the forward method. Handle optional mask.
        .def("forward", [](TransformerBlockImpl &self, torch::Tensor x, c10::optional<torch::Tensor> mask) {
            return self.forward(x, mask);
        }, pybind11::arg("x"), pybind11::arg("attention_mask") = pybind11::none(), "Forward pass for TransformerBlock");

    // Bind the custom LayerNorm function (for demonstration/testing if needed)
    m.def("layer_norm_custom_cpp", &layer_norm_custom, "Custom Layer Normalization forward (uses custom backward)");

    // Bind illustrative LayerNorm forward/backward C++ helpers (optional, for testing/understanding)
     m.def("layer_norm_forward_cpp", &layer_norm_forward, "Illustrative Layer Normalization Forward C++");
     m.def("layer_norm_backward_cpp", &layer_norm_backward, "Illustrative Layer Normalization Backward C++");

}
