import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from typing import Optional, Tuple

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 8192):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        
        # Generate and cache the rotation matrices
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        position = torch.arange(max_position_embeddings).float()
        sinusoid_inp = torch.einsum("i,j->ij", position, inv_freq)
        self.register_buffer("sin", sinusoid_inp.sin())
        self.register_buffer("cos", sinusoid_inp.cos())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # x: [batch_size, seq_len, dim]
        sin = self.sin[:seq_len].view(1, seq_len, 1, self.dim // 2)
        cos = self.cos[:seq_len].view(1, seq_len, 1, self.dim // 2)
        
        # Reshape for broadcasting
        x1, x2 = x.chunk(2, dim=-1)
        x1 = x1.view(x1.shape[0], x1.shape[1], 1, -1)
        x2 = x2.view(x2.shape[0], x2.shape[1], 1, -1)
        
        # Apply rotation
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        # Reshape back to original format
        return rotated.view(x.shape[0], x.shape[1], -1)

class MultiHeadLatentAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        latent_dim: int,
        dropout: float = 0.1,
        max_position_embeddings: int = 8192
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.latent_dim = latent_dim
        self.head_dim = hidden_size // num_attention_heads
        
        # Latent compression matrices
        self.W_DKV = nn.Linear(hidden_size, latent_dim)
        self.W_DQ = nn.Linear(hidden_size, latent_dim)
        
        # Uncompression matrices
        self.W_UK = nn.Linear(latent_dim, hidden_size)
        self.W_UV = nn.Linear(latent_dim, hidden_size)
        self.W_UQ = nn.Linear(latent_dim, hidden_size)
        
        # Position-specific matrices
        self.W_KR = nn.Linear(hidden_size, self.head_dim)
        self.W_QR = nn.Linear(hidden_size, self.head_dim)
        
        # Output projection
        self.W_O = nn.Linear(hidden_size, hidden_size)
        
        # RoPE for positional encoding
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_position_embeddings)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Generate latent vectors
        c_kv = self.W_DKV(hidden_states)  # [batch_size, seq_len, latent_dim]
        c_q = self.W_DQ(hidden_states)    # [batch_size, seq_len, latent_dim]
        
        # Generate compressed K, V, Q
        k_c = self.W_UK(c_kv)  # [batch_size, seq_len, hidden_size]
        v_c = self.W_UV(c_kv)  # [batch_size, seq_len, hidden_size]
        q_c = self.W_UQ(c_q)   # [batch_size, seq_len, hidden_size]
        
        # Generate position-specific K, Q
        k_r = self.W_KR(hidden_states)     # [batch_size, seq_len, head_dim]
        q_r = self.W_QR(hidden_states)     # [batch_size, seq_len, head_dim]
        
        # Apply RoPE
        k_r = self.rope(k_r, seq_len)      # [batch_size, seq_len, head_dim]
        q_r = self.rope(q_r, seq_len)      # [batch_size, seq_len, head_dim]
        
        # Reshape for multi-head attention
        k_c = rearrange(k_c, 'b s (h d) -> b h s d', h=self.num_attention_heads)
        v_c = rearrange(v_c, 'b s (h d) -> b h s d', h=self.num_attention_heads)
        q_c = rearrange(q_c, 'b s (h d) -> b h s d', h=self.num_attention_heads)
        
        # Reshape position-specific components for multi-head attention
        k_r = k_r.unsqueeze(1).expand(batch_size, self.num_attention_heads, seq_len, self.head_dim)
        q_r = q_r.unsqueeze(1).expand(batch_size, self.num_attention_heads, seq_len, self.head_dim)
        
        # Combine compressed and position-specific components
        k = torch.cat([k_c, k_r], dim=-1)
        q = torch.cat([q_c, q_r], dim=-1)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            # Expand attention mask for multi-head attention
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v_c)
        context = rearrange(context, 'b h s d -> b s (h d)')
        
        # Final projection
        output = self.W_O(context)
        
        if output_attentions:
            return output, attn_weights
        return output, None 