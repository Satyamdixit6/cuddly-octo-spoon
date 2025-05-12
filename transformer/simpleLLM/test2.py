import torch
import transformer_cpp

# Example inputs
batch_size, seq_len, embedding_dim, num_heads = 2, 8, 256, 8
query = torch.randn(batch_size, seq_len, embedding_dim).cuda()
key = torch.randn(batch_size, seq_len, embedding_dim).cuda()
value = torch.randn(batch_size, seq_len, embedding_dim).cuda()

# Mask (optional)
mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).unsqueeze(0).unsqueeze(0).cuda()

# Compute attention
output = transformer_cpp.multi_head_self_attention(query, key, value, mask, num_heads)
print("Output shape:", output.shape)