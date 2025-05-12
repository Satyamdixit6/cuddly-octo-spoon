import torch
import torch.nn as nn

class OutputProcessor(nn.Module):
    """
    Handles output projection and token prediction.
    """
    def __init__(self, embedding_dim, vocab_size):
        """
        Args:
            embedding_dim (int): Dimension of token embeddings.
            vocab_size (int): Size of the vocabulary.
        """
        super(OutputProcessor, self).__init__()
        self.output_projection = nn.Linear(embedding_dim, vocab_size)

    def forward(self, transformer_output):
        """
        Forward pass to compute logits for each token.
        """
        logits = self.output_projection(transformer_output)  # Shape: (batch_size, seq_len, vocab_size)
        return logits


# Example usage
if __name__ == "__main__":
    # Example inputs
    batch_size, seq_len, embedding_dim, vocab_size = 2, 8, 256, 30522
    transformer_output = torch.randn(batch_size, seq_len, embedding_dim).cuda()

    # Initialize output processor
    output_processor = OutputProcessor(embedding_dim, vocab_size).cuda()

    # Compute logits
    logits = output_processor(transformer_output)
    print("Logits shape:", logits.shape)

    # Predict next token
    predicted_tokens = torch.argmax(logits, dim=-1)
    print("Predicted tokens:", predicted_tokens)