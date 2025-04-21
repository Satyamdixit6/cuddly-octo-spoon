import torch
import torch.nn as nn
# import transformers_cpp

class OutputProcessor(nn.Module):  # Fixed class name and colon
    def __init__(self, embedding_dim, vocab_size):  # Fixed colon
        super(OutputProcessor, self).__init__()
        self.output_projection = nn.Linear(embedding_dim, vocab_size)

    def forward(self, moe_output):
        logits = self.output_projection(moe_output)  # Fixed spacing
        return logits


