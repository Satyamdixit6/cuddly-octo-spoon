import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class Expert(nn.Module):
    def __init__(self, hidden_size: int, expert_hidden_size: int, activation_fn: str = "gelu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_hidden_size = expert_hidden_size
        
        self.up_proj = nn.Linear(hidden_size, expert_hidden_size)
        self.down_proj = nn.Linear(expert_hidden_size, hidden_size)
        
        if activation_fn == "gelu":
            self.activation = F.gelu
        elif activation_fn == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.up_proj(x)))

class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_selected_experts: int,
        expert_hidden_size: int,
        shared_expert_size: int = 1,
        activation_fn: str = "gelu",
        load_balancing_weight: float = 0.01,
        bias_update_speed: float = 0.01
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_selected_experts = num_selected_experts
        self.expert_hidden_size = expert_hidden_size
        self.shared_expert_size = shared_expert_size
        self.load_balancing_weight = load_balancing_weight
        self.bias_update_speed = bias_update_speed
        
        # Router
        self.router = nn.Linear(hidden_size, num_experts)
        
        # Expert bias for load balancing
        self.register_buffer("expert_bias", torch.zeros(num_experts))
        
        # Experts
        self.experts = nn.ModuleList([
            Expert(hidden_size, expert_hidden_size, activation_fn)
            for _ in range(num_experts)
        ])
        
        # Shared experts
        self.shared_experts = nn.ModuleList([
            Expert(hidden_size, expert_hidden_size, activation_fn)
            for _ in range(shared_expert_size)
        ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_router_logits: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Calculate routing scores
        router_logits = self.router(hidden_states)  # [batch_size, seq_len, num_experts]
        
        # Add load balancing bias
        if self.training:
            router_logits = router_logits + self.expert_bias
        
        # Calculate expert weights
        expert_weights = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(
            expert_weights,
            self.num_selected_experts,
            dim=-1
        )
        
        # Process through shared experts
        shared_output = sum(
            expert(hidden_states)
            for expert in self.shared_experts
        )
        
        # Process through selected experts
        expert_outputs = []
        for i in range(self.num_selected_experts):
            expert_idx = top_k_indices[..., i]
            expert_output = torch.zeros_like(hidden_states)
            
            # Process each expert's tokens
            for j in range(self.num_experts):
                mask = (expert_idx == j)
                if mask.any():
                    expert_output[mask] = self.experts[j](hidden_states[mask])
            
            expert_outputs.append(expert_output * top_k_weights[..., i:i+1])
        
        # Combine expert outputs
        expert_output = sum(expert_outputs)
        
        # Update expert bias for load balancing
        if self.training:
            with torch.no_grad():
                # Calculate expert usage
                expert_usage = torch.zeros(self.num_experts, device=hidden_states.device)
                for i in range(self.num_selected_experts):
                    expert_idx = top_k_indices[..., i]
                    for j in range(self.num_experts):
                        expert_usage[j] += (expert_idx == j).float().mean()
                
                # Update bias
                target_usage = self.num_selected_experts / self.num_experts
                self.expert_bias += self.bias_update_speed * (expert_usage - target_usage)
        
        # Combine shared and expert outputs
        output = hidden_states + shared_output + expert_output
        
        if output_router_logits:
            return output, router_logits
        return output, None
    
    def get_load_balancing_loss(self) -> torch.Tensor:
        """Calculate the load balancing loss."""
        if not self.training:
            return torch.tensor(0.0, device=self.expert_bias.device)
        
        # Calculate expert usage
        expert_usage = torch.zeros(self.num_experts, device=self.expert_bias.device)
        for i in range(self.num_selected_experts):
            expert_idx = self.top_k_indices[..., i]
            for j in range(self.num_experts):
                expert_usage[j] += (expert_idx == j).float().mean()
        
        # Calculate load balancing loss
        target_usage = self.num_selected_experts / self.num_experts
        load_balancing_loss = self.load_balancing_weight * torch.sum(
            expert_usage * torch.log(expert_usage / target_usage)
        )
        
        return load_balancing_loss 