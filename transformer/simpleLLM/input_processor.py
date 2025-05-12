import torch
import torch.nn as nn
from transformers import AutoTokenizer

class TextDataset(torch.utils.data.Dataset):
    """
    Custom Dataset class to tokenize input text and prepare it for training.
    """
    def __init__(self, texts, tokenizer, max_seq_len=512):
        """
        Args:
            texts (list of str): List of input sentences or documents.
            tokenizer: Pre-trained tokenizer (e.g., Hugging Face tokenizer).
            max_seq_len (int): Maximum sequence length for padding/truncation.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Tokenize a single input text and prepare inputs and targets.
        """
        text = self.texts[idx]
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].squeeze(0)  # Remove batch dimension
        target_ids = torch.roll(input_ids, shifts=-1, dims=0)  # Shifted targets
        target_ids[-1] = self.tokenizer.pad_token_id  # Replace last token with padding
        return input_ids, target_ids


def collate_fn(batch):
    """
    Collate function to stack input_ids and target_ids into batches.
    """
    input_ids, target_ids = zip(*batch)
    input_ids = torch.stack(input_ids)
    target_ids = torch.stack(target_ids)
    return input_ids, target_ids


class InputProcessor(nn.Module):
    """
    Handles token embedding and positional encoding.
    """
    def __init__(self, vocab_size, embedding_dim, max_seq_len=512):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of token embeddings.
            max_seq_len (int): Maximum sequence length for positional encoding.
        """
        super(InputProcessor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = self._create_positional_encoding(max_seq_len, embedding_dim)

    @staticmethod
    def _create_positional_encoding(max_seq_len, embedding_dim):
        """
        Create sinusoidal positional encodings.
        """
        position = torch.arange(max_seq_len).unsqueeze(1)  # Shape: (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe = torch.zeros(max_seq_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # Shape: (1, max_seq_len, embedding_dim)

    def forward(self, input_ids):
        """
        Forward pass to compute token embeddings and add positional encodings.
        """
        embeddings = self.embedding(input_ids)  # Shape: (batch_size, seq_len, embedding_dim)
        seq_len = input_ids.size(1)
        positional_encoding = self.positional_encoding[:, :seq_len, :].to(input_ids.device)
        return embeddings + positional_encoding


# Example usage
if __name__ == "__main__":
    # Example data
    texts = [
        "The capital of France is Paris.",
        "Transformers are powerful models.",
        "Machine learning is fascinating."
    ]

    # Load pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Create dataset and dataloader
    dataset = TextDataset(texts, tokenizer, max_seq_len=32)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # Initialize input processor
    vocab_size = tokenizer.vocab_size
    embedding_dim = 256
    input_processor = InputProcessor(vocab_size, embedding_dim, max_seq_len=32)

    # Process a batch of inputs
    for batch in dataloader:
        input_ids, target_ids = batch
        embeddings = input_processor(input_ids)
        print("Embeddings shape:", embeddings.shape)
        break


