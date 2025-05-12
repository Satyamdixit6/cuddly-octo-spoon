import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from input_processor import TextDataset, collate_fn
from transformer_pipeline import TransformerPipeline

def train(model, dataloader, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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


def infer(model, input_text, tokenizer, max_length=50):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()

        for _ in range(max_length):
            logits = model(input_ids)
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

        output_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return output_text


# Example usage
if __name__ == "__main__":
    # Load tokenizer and dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    texts = ["The capital of France is Paris.", "Transformers are powerful models."]
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # Hyperparameters
    vocab_size = tokenizer.vocab_size
    embedding_dim = 256
    num_layers = 2
    num_heads_list = [8, 8]
    num_experts_list = [4, 4]
    expert_hidden_dim = 512

    # Initialize model
    model = TransformerPipeline(vocab_size, embedding_dim, num_layers, num_heads_list, num_experts_list, expert_hidden_dim).cuda()

    # Train the model
    train(model, dataloader, epochs=3, lr=1e-4)

    # Inference
    input_text = "The capital of France is"
    output_text = infer(model, input_text, tokenizer)
    print("Generated text:", output_text)
    