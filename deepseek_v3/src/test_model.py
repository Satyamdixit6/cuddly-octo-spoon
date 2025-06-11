import torch
import sys
import os
import gc

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.model_config import DeepSeekConfig
from src.model.model import DeepSeekV3

def test_model():
    print("Testing DeepSeek-V3 model...")
    
    # Create a minimal config for testing
    config = DeepSeekConfig(
        vocab_size=1000,  # Smaller vocabulary for testing
        hidden_size=128,  # Even smaller hidden size
        num_hidden_layers=2,  # Fewer layers
        num_attention_heads=4,
        latent_dim=32,
        num_experts=2,
        num_selected_experts=1,
        expert_hidden_size=256,
        shared_expert_size=1,
        num_mtp_layers=1,
        max_position_embeddings=128,
        dropout=0.1,
        attention_dropout=0.1,
        activation_function="gelu",
        load_balancing_weight=0.01,
        bias_update_speed=0.01,
        mtp_weight=0.1
    )

    try:
        # Enable memory optimization
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        # Initialize model
        print("Initializing model...")
        model = DeepSeekV3(config)
        model.eval()

        # Move model to CPU if CUDA is not available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Create test input
        print("Creating test input...")
        batch_size = 1
        seq_length = 8
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length), device=device)
        attention_mask = torch.ones((batch_size, seq_length), device=device)

        # Forward pass
        print("Running forward pass...")
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False
            )

        # Print output shapes
        print("\nTest successful! Output shapes:")
        print(f"Main logits shape: {outputs['logits'].shape}")
        print(f"MTP outputs shape: {[out.shape for out in outputs['mtp_outputs']]}")
        print(f"MTP loss: {outputs['mtp_loss']}")

        # Clean up
        del model
        del outputs
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    test_model() 