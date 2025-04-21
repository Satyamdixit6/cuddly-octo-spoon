import torch
import torch.nn as nn
from transformers import AutoTokenizer

class InputProcessor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len=512):
        super(InputProcessor, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Pre-trained tokenizer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = self._create_positional_encoding(max_seq_len, embedding_dim)

    @staticmethod
    def _create_positional_encoding(max_seq_len, embedding_dim):
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe = torch.zeros(max_seq_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, max_seq_len, embedding_dim)

    def forward(self, input_text):
        # Tokenize input text
        tokenized = self.tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
        input_ids = tokenized["input_ids"].cuda()  # Shape: (batch_size, seq_len)

        # Token embedding
        token_embeddings = self.embedding(input_ids)  # Shape: (batch_size, seq_len, embedding_dim)

        # Positional encoding
        seq_len = input_ids.size(1)
        positional_encoding = self.positional_encoding[:, :seq_len, :].to(input_ids.device)
        embeddings = token_embeddings + positional_encoding

        return embeddings, input_ids