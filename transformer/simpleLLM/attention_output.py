import torch
import transformer_cpp

# Test Multi-Head Attention
batch_size, seq_len, embedding_dim, num_heads = 2, 8, 256, 8
query = torch.randn(batch_size, seq_len, embedding_dim).cuda()
key = torch.randn(batch_size, seq_len, embedding_dim).cuda()
value = torch.randn(batch_size, seq_len, embedding_dim).cuda()
mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).cuda()

output = transformer_cpp.multi_head_self_attention(query, key, value, mask, num_heads)
print("Attention Output Shape:", output.shape)

# Test MoE
input = torch.randn(batch_size, seq_len, embedding_dim).cuda()
expert_weights = [torch.randn(embedding_dim, 512).cuda() for _ in range(4)]
expert_biases = [torch.randn(512).cuda() for _ in range(4)]
moe_output = transformer_cpp.mixture_of_experts(input, expert_weights, expert_biases, k=2)
print("MoE Output Shape:", moe_output.shape)

# Test Transformer Model
num_heads_list = [8, 8]
num_experts_list = [4, 4]
expert_hidden_dim = 512
transformer_output = transformer_cpp.transformer_model(input, num_heads_list, num_experts_list, expert_hidden_dim)
print("Transformer Output Shape:", transformer_output.shape)