import torch
from config.model_config import DeepSeekConfig
from model.model import DeepSeekV3

def main():
    # Create model configuration with lighter parameters
    config = DeepSeekConfig(
        vocab_size=32000,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        latent_dim=64,
        num_experts=4,
        num_selected_experts=2,
        expert_hidden_size=1024,
        shared_expert_size=1,
        num_mtp_layers=2,
        max_position_embeddings=2048,
        dropout=0.1,
        attention_dropout=0.1,
        activation_function="gelu",
        load_balancing_weight=0.01,
        bias_update_speed=0.01,
        mtp_weight=0.1
    )

    # Initialize model
    model = DeepSeekV3(config)
    model.eval()  # Set to evaluation mode

    # Create example input with smaller batch size and sequence length
    batch_size = 2
    seq_length = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )

    # Print output shapes
    print(f"Main logits shape: {outputs['logits'].shape}")
    print(f"MTP outputs shape: {[out.shape for out in outputs['mtp_outputs']]}")
    print(f"MTP loss: {outputs['mtp_loss']}")
    print(f"Number of hidden states: {len(outputs['hidden_states']) if outputs['hidden_states'] else 0}")
    print(f"Number of attention layers: {len(outputs['attentions']) if outputs['attentions'] else 0}")

if __name__ == "__main__":
    main() 