import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset  # Hugging Face datasets
from input_processor import InputProcessor
from output_processor import OutputProcessor
from transformer_pipeline import TransformerPipeline

# Step 1: Load and preprocess the dataset
def load_data(batch_size=32, max_seq_len=32):
    # Load Wikipedia dataset from Hugging Face
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:1%]")  # Use 1% of the dataset for small-scale training

    # Tokenizer
    tokenizer = lambda text: text.lower().split()  # Simple tokenizer

    # Build vocabulary
    def yield_tokens(dataset):
        for example in dataset:
            if example["text"].strip():  # Skip empty lines
                yield tokenizer(example["text"])

    vocab = set()
    for tokens in yield_tokens(dataset):
        vocab.update(tokens)
    vocab = ["<unk>", "<pad>", "<bos>", "<eos>"] + list(vocab)
    vocab = {word: idx for idx, word in enumerate(vocab)}

    # Convert text to token IDs
    def text_to_ids(text):
        tokens = tokenizer(text)
        token_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens][:max_seq_len]
        token_ids = [vocab["<bos>"]] + token_ids + [vocab["<eos>"]]
        return token_ids + [vocab["<pad>"]] * (max_seq_len - len(token_ids))

    # Prepare dataset
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            text = self.dataset[idx]["text"]
            input_ids = text_to_ids(text)
            target_ids = input_ids[1:] + [vocab["<pad>"]]  # Shifted targets
            return torch.tensor(input_ids), torch.tensor(target_ids)

    def collate_fn(batch):
        input_ids, target_ids = zip(*batch)
        input_ids = torch.stack(input_ids)
        target_ids = torch.stack(target_ids)
        return input_ids, target_ids

    # Create DataLoader
    train_dataset = TextDataset(dataset)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader, vocab

# Step 2: Training function
def train(model, dataloader, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])  # Ignore padding tokens

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids, target_ids = batch
            input_ids, target_ids = input_ids.cuda(), target_ids.cuda()

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")

# Step 3: Inference function
def infer(model, input_text, tokenizer, vocab, max_length=50):
    model.eval()
    with torch.no_grad():
        tokens = tokenizer(input_text)
        token_ids = [vocab["<bos>"]] + [vocab.get(token, vocab["<unk>"]) for token in tokens] + [vocab["<eos>"]]
        input_ids = torch.tensor([token_ids]).cuda()

        for _ in range(max_length):
            logits = model(input_ids)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            if next_token_id.item() == vocab["<eos>"]:
                break

        output_ids = input_ids[0].cpu().tolist()
        output_text = " ".join([list(vocab.keys())[list(vocab.values()).index(idx)] for idx in output_ids if idx not in [vocab["<bos>"], vocab["<pad>"]]])
        return output_text

# Step 4: Main script
if __name__ == "__main__":
    # Load dataset and vocabulary
    dataloader, vocab = load_data(batch_size=32, max_seq_len=32)

    # Hyperparameters
    vocab_size = len(vocab)
    embedding_dim = 128
    num_layers = 2
    num_heads_list = [4, 4]
    num_experts_list = [2, 2]
    expert_hidden_dim = 256

    # Initialize model
    model = TransformerPipeline(vocab_size, embedding_dim, num_layers, num_heads_list, num_experts_list, expert_hidden_dim).cuda()

    # Train the model
    train(model, dataloader, epochs=3, lr=1e-4)

    # Inference
    tokenizer = lambda text: text.lower().split()  # Simple tokenizer
    input_text = "The capital of France is"
    output_text = infer(model, input_text, tokenizer, vocab)
    print("Generated text:", output_text)