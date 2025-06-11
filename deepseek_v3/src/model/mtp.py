import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .moe import MixtureOfExperts

class MTPModule(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_layers: int,
        num_attention_heads: int,
        latent_dim: int,
        num_experts: int,
        num_selected_experts: int,
        expert_hidden_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Projection matrix for combining representations
        self.projection = nn.Linear(hidden_size * 2, hidden_size)
        
        # Transformer block
        self.transformer_block = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.MultiheadAttention(
                    hidden_size,
                    num_attention_heads,
                    dropout=dropout,
                    batch_first=True
                ),
                'moe': MixtureOfExperts(
                    hidden_size,
                    num_experts,
                    num_selected_experts,
                    expert_hidden_size
                ),
                'norm1': nn.LayerNorm(hidden_size),
                'norm2': nn.LayerNorm(hidden_size)
            })
            for _ in range(num_layers)
        ])
        
        # Output head
        self.output_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Ensure both inputs have the same sequence length
        seq_len = min(hidden_states.size(1), token_embeddings.size(1))
        hidden_states = hidden_states[:, :seq_len, :]
        token_embeddings = token_embeddings[:, :seq_len, :]
        
        # Combine representations
        combined = self.projection(
            torch.cat([
                F.layer_norm(hidden_states, [self.hidden_size]),
                F.layer_norm(token_embeddings, [self.hidden_size])
            ], dim=-1)
        )
        
        # Process through transformer blocks
        for block in self.transformer_block:
            # Attention
            attn_output, _ = block['attention'](
                block['norm1'](combined),
                block['norm1'](combined),
                block['norm1'](combined),
                key_padding_mask=attention_mask[:, :seq_len] if attention_mask is not None else None
            )
            combined = combined + attn_output
            
            # MoE
            moe_output, _ = block['moe'](block['norm2'](combined))
            combined = combined + moe_output
        
        # Generate predictions
        logits = self.output_head(combined)
        return logits

class MultiTokenPrediction(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_layers: int,
        num_attention_heads: int,
        latent_dim: int,
        num_experts: int,
        num_selected_experts: int,
        expert_hidden_size: int,
        num_mtp_layers: int = 2,
        mtp_weight: float = 0.1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_mtp_layers = num_mtp_layers
        self.mtp_weight = mtp_weight
        
        # MTP modules
        self.mtp_modules = nn.ModuleList([
            MTPModule(
                hidden_size,
                vocab_size,
                num_layers,
                num_attention_heads,
                latent_dim,
                num_experts,
                num_selected_experts,
                expert_hidden_size,
                dropout
            )
            for _ in range(num_mtp_layers)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            token_embeddings: List of [batch_size, seq_len, hidden_size] tensors
            attention_mask: [batch_size, seq_len]
        """
        mtp_outputs = []
        mtp_losses = []
        
        for i, (module, token_emb) in enumerate(zip(self.mtp_modules, token_embeddings)):
            # Ensure token embeddings have the correct shape
            if token_emb.size(1) > hidden_states.size(1):
                token_emb = token_emb[:, :hidden_states.size(1), :]
            
            # Generate predictions
            logits = module(hidden_states, token_emb, attention_mask)
            mtp_outputs.append(logits)
            
            # Calculate loss
            if self.training:
                # Shift token embeddings for next token prediction
                target_tokens = token_emb[:, 1:]
                pred_logits = logits[:, :-1]
                
                # Ensure shapes match for loss calculation
                if target_tokens.size(1) != pred_logits.size(1):
                    min_len = min(target_tokens.size(1), pred_logits.size(1))
                    target_tokens = target_tokens[:, :min_len]
                    pred_logits = pred_logits[:, :min_len]
                
                # Calculate cross entropy loss
                loss = F.cross_entropy(
                    pred_logits.reshape(-1, self.vocab_size),
                    target_tokens.reshape(-1),
                    reduction='mean'
                )
                mtp_losses.append(loss)
        
        # Combine losses
        if self.training and mtp_losses:
            total_mtp_loss = self.mtp_weight * sum(mtp_losses) / len(mtp_losses)
        else:
            total_mtp_loss = torch.tensor(0.0, device=hidden_states.device)
        
        return mtp_outputs, total_mtp_loss 