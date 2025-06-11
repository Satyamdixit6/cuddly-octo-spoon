from dataclasses import dataclass
from typing import Optional

@dataclass
class DeepSeekConfig:
    # Model dimensions
    vocab_size: int = 32000
    hidden_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 1024
    
    # MLA specific
    latent_dim: int = 64
    position_embedding_type: str = "rotary"
    max_position_embeddings: int = 2048
    
    # MoE specific
    num_experts: int = 4
    num_selected_experts: int = 2
    expert_hidden_size: int = 1024
    shared_expert_size: int = 1
    
    # Training specific
    dropout: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    activation_function: str = "gelu"
    
    # MTP specific
    num_mtp_layers: int = 2
    mtp_weight: float = 0.1
    
    # Load balancing
    load_balancing_weight: float = 0.01
    bias_update_speed: float = 0.01
    
    # Quantization
    use_fp8: bool = False
    fp8_quantization_granularity: str = "tile"
    fp8_accumulation_interval: int = 128
    
    # Parallelism
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    data_parallel_size: int = 1
    
    # Inference
    max_batch_size: int = 4
    max_sequence_length: int = 512
    
    def __post_init__(self):
        # Validate configurations
        assert self.hidden_size % self.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        assert self.latent_dim < self.hidden_size, "latent_dim must be smaller than hidden_size"
        assert self.num_selected_experts <= self.num_experts, "num_selected_experts must be less than or equal to num_experts" 