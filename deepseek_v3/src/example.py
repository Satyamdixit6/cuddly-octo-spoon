import torch
from model import DeepSeekV3

def main():
    # Model parameters - reduced for laptop compatibility
    vocab_size = 10000  # Reduced vocabulary size
    dim = 512          # Reduced model dimension
    num_layers = 4     # Fewer layers
    num_heads = 8      # Fewer attention heads
    latent_dim = 64    # Smaller latent dimension
    num_experts = 4    # Fewer experts
    num_selected = 2   # Keep same number of selected experts
    hidden_dim = 1024  # Smaller hidden dimension
    
    # Create model
    model = DeepSeekV3(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        latent_dim=latent_dim,
        num_experts=num_experts,
        num_selected=num_selected,
        hidden_dim=hidden_dim
    )
    
    # Example input - smaller batch and sequence length
    batch_size = 1
    seq_len = 8
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Calculate number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Print memory usage
    print(f"Model size in MB: {total_params * 4 / (1024 * 1024):.2f}")  # Assuming float32

if __name__ == "__main__":
    main() 