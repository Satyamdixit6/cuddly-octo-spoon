import torch
import torch.nn as nn
import transformer_cpp
from input_processor import InputProcessor
from output_processor import OutputProcessor

class TransformerPipeline(nn.Module):
    """
    Full Transformer pipeline integrating input processing, CUDA-based Transformer, and output decoding.
    """
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads_list, num_experts_list, expert_hidden_dim, max_seq_len=512):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of token embeddings.
            num_layers (int): Number of Transformer layers.
            num_heads_list (list of int): Number of attention heads per layer.
            num_experts_list (list of int): Number of experts per layer.
            expert_hidden_dim (int): Hidden dimension of each expert.
            max_seq_len (int): Maximum sequence length for positional encoding.
        """
        super(TransformerPipeline, self).__init__()
        self.input_processor = InputProcessor(vocab_size, embedding_dim, max_seq_len)
        self.output_processor = OutputProcessor(embedding_dim, vocab_size)
        self.num_layers = num_layers
        self.num_heads_list = num_heads_list
        self.num_experts_list = num_experts_list
        self.expert_hidden_dim = expert_hidden_dim

    def forward(self, input_ids):
        """
        Forward pass through the full Transformer pipeline.
        """
        # Step 1: Input processing (token embedding + positional encoding)
        embeddings = self.input_processor(input_ids)

        # Step 2: CUDA-based Transformer
        transformer_output = transformer_cpp.transformer_model(
            embeddings,
            self.num_heads_list,
            self.num_experts_list,
            self.expert_hidden_dim
        )

        # Step 3: Output processing (logits computation)
        logits = self.output_processor(transformer_output)

        return logits


# Example usage
if __name__ == "__main__":
    # Example inputs
    batch_size, seq_len, vocab_size, embedding_dim = 2, 8, 30522, 256
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()

    # Hyperparameters
    num_layers = 2
    num_heads_list = [8, 8]
    num_experts_list = [4, 4]
    expert_hidden_dim = 512

    # Initialize Transformer pipeline
    model = TransformerPipeline(vocab_size, embedding_dim, num_layers, num_heads_list, num_experts_list, expert_hidden_dim).cuda()

    # Forward pass
    logits = model(input_ids)
    print("Logits shape:", logits.shape)

    # Predict next token
    predicted_tokens = torch.argmax(logits, dim=-1)
    print("Predicted tokens:", predicted_tokens)