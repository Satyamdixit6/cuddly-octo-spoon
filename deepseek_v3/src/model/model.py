import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any

from .mla import MultiHeadLatentAttention
from .moe import MixtureOfExperts
from .mtp import MultiTokenPrediction
from ..config.model_config import DeepSeekConfig

class DeepSeekV3Block(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        
        # Attention block
        self.attention = MultiHeadLatentAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            latent_dim=config.latent_dim,
            dropout=config.attention_dropout,
            max_position_embeddings=config.max_position_embeddings
        )
        
        # MoE block
        self.moe = MixtureOfExperts(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            num_selected_experts=config.num_selected_experts,
            expert_hidden_size=config.expert_hidden_size,
            shared_expert_size=config.shared_expert_size,
            activation_fn=config.activation_function,
            load_balancing_weight=config.load_balancing_weight,
            bias_update_speed=config.bias_update_speed
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Attention block
        attn_output, attn_weights = self.attention(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        hidden_states = hidden_states + self.dropout(attn_output)
        
        # MoE block
        moe_output, _ = self.moe(self.norm2(hidden_states))
        hidden_states = hidden_states + self.dropout(moe_output)
        
        if output_attentions:
            return hidden_states, attn_weights
        return hidden_states, None

class DeepSeekV3(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embeddings
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DeepSeekV3Block(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size)
        
        # Output head
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # MTP module
        self.mtp = MultiTokenPrediction(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            num_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            latent_dim=config.latent_dim,
            num_experts=config.num_experts,
            num_selected_experts=config.num_selected_experts,
            expert_hidden_size=config.expert_hidden_size,
            num_mtp_layers=config.num_mtp_layers,
            mtp_weight=config.mtp_weight,
            dropout=config.dropout
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> Dict[str, Any]:
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        token_embeddings = self.embedding(input_ids)
        
        # Get position embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeddings = self.position_embedding(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeddings + position_embeddings
        
        # Process through transformer blocks
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            hidden_states, attention_weights = block(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions
            )
            
            if output_attentions:
                all_attentions.append(attention_weights)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Generate main model predictions
        logits = self.head(hidden_states)
        
        # Prepare token embeddings for MTP
        mtp_token_embeddings = []
        for i in range(self.config.num_mtp_layers):
            # Shift token embeddings for next token prediction
            shifted_embeddings = token_embeddings[:, i+1:]
            if shifted_embeddings.size(1) > 0:  # Only add if there are tokens left
                mtp_token_embeddings.append(shifted_embeddings)
        
        # Generate MTP predictions
        mtp_outputs, mtp_loss = self.mtp(
            hidden_states,
            mtp_token_embeddings,
            attention_mask
        )
        
        if not return_dict:
            return (logits, mtp_outputs, mtp_loss, all_hidden_states, all_attentions)
        
        return {
            "logits": logits,
            "mtp_outputs": mtp_outputs,
            "mtp_loss": mtp_loss,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions
        }
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values
        }
    
    def _reorder_cache(
        self,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        beam_idx: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        reordered_past = []
        for layer_past in past_key_values:
            reordered_past.append(
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            )
        return reordered_past 