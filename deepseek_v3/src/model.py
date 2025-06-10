import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class RoPE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x, seq_len):
        # Simplified RoPE implementation
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=x.device) * -(math.log(10000.0) / self.dim))
        pe = torch.zeros(seq_len, self.dim, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x * pe.unsqueeze(0)

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, dim, num_heads, latent_dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = dim // num_heads
        
        # Latent compression matrices
        self.W_DKV = nn.Linear(dim, latent_dim)
        self.W_DQ = nn.Linear(dim, latent_dim)
        
        # Uncompression matrices
        self.W_UK = nn.Linear(latent_dim, dim)
        self.W_UV = nn.Linear(latent_dim, dim)
        self.W_UQ = nn.Linear(latent_dim, dim)
        
        # Position-specific matrices
        self.W_KR = nn.Linear(dim, self.head_dim)
        self.W_QR = nn.Linear(dim, self.head_dim)
        
        # Output projection
        self.W_O = nn.Linear(dim, dim)
        
        # RoPE for positional encoding
        self.rope = RoPE(self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Generate latent vectors
        c_kv = self.W_DKV(x)  # [batch_size, seq_len, latent_dim]
        c_q = self.W_DQ(x)    # [batch_size, seq_len, latent_dim]
        
        # Generate compressed K, V, Q
        k_c = self.W_UK(c_kv)  # [batch_size, seq_len, dim]
        v_c = self.W_UV(c_kv)  # [batch_size, seq_len, dim]
        q_c = self.W_UQ(c_q)   # [batch_size, seq_len, dim]
        
        # Generate position-specific K, Q
        k_r = self.W_KR(x)     # [batch_size, seq_len, head_dim]
        q_r = self.W_QR(x)     # [batch_size, seq_len, head_dim]
        
        # Apply RoPE
        k_r = self.rope(k_r, seq_len)
        q_r = self.rope(q_r, seq_len)
        
        # Reshape for multi-head attention
        k_c = rearrange(k_c, 'b s (h d) -> b h s d', h=self.num_heads)
        v_c = rearrange(v_c, 'b s (h d) -> b h s d', h=self.num_heads)
        q_c = rearrange(q_c, 'b s (h d) -> b h s d', h=self.num_heads)
        
        # Combine compressed and position-specific components
        k = torch.cat([k_c, k_r.unsqueeze(1).expand(-1, self.num_heads, -1, -1)], dim=-1)
        q = torch.cat([q_c, q_r.unsqueeze(1).expand(-1, self.num_heads, -1, -1)], dim=-1)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v_c)
        context = rearrange(context, 'b h s d -> b s (h d)')
        
        # Final projection
        output = self.W_O(context)
        
        return output

class Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, x):
        return self.net(x)

class MixtureOfExperts(nn.Module):
    def __init__(self, dim, num_experts, num_selected, hidden_dim):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_selected = num_selected
        
        # Router
        self.router = nn.Linear(dim, num_experts)
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(dim, hidden_dim) for _ in range(num_experts)
        ])
        
        # Shared expert
        self.shared_expert = Expert(dim, hidden_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Calculate routing scores
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]
        
        # Add load balancing bias during training
        if self.training:
            router_logits = router_logits + torch.randn_like(router_logits) * 0.1
        
        # Select top-k experts
        expert_weights = F.softmax(router_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(expert_weights, self.num_selected, dim=-1)
        
        # Process through shared expert
        shared_output = self.shared_expert(x)
        
        # Process through selected experts
        expert_outputs = []
        for i in range(self.num_selected):
            expert_idx = top_k_indices[..., i]
            expert_output = torch.zeros_like(x)
            
            # Process each expert's tokens
            for j in range(self.num_experts):
                mask = (expert_idx == j)
                if mask.any():
                    expert_output[mask] = self.experts[j](x[mask])
            
            expert_outputs.append(expert_output * top_k_weights[..., i:i+1])
        
        # Combine expert outputs
        expert_output = sum(expert_outputs)
        
        # Combine shared and expert outputs
        return shared_output + expert_output

class DeepSeekV3Block(nn.Module):
    def __init__(self, dim, num_heads, latent_dim, num_experts, num_selected, hidden_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadLatentAttention(dim, num_heads, latent_dim, dropout)
        self.moe = MixtureOfExperts(dim, num_experts, num_selected, hidden_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x, mask=None):
        # Attention block
        attn_output = self.attention(self.norm1(x), mask)
        x = x + attn_output
        
        # MoE block
        moe_output = self.moe(self.norm2(x))
        x = x + moe_output
        
        return x

class DeepSeekV3(nn.Module):
    def __init__(self, 
                 vocab_size,
                 dim=2048,
                 num_layers=32,
                 num_heads=32,
                 latent_dim=256,
                 num_experts=8,
                 num_selected=2,
                 hidden_dim=4096,
                 dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            DeepSeekV3Block(dim, num_heads, latent_dim, num_experts, num_selected, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        
        for block in self.blocks:
            x = block(x, mask)
            
        x = self.norm(x)
        logits = self.head(x)
        
        return logits 