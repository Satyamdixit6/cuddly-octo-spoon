import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as L
from transformer_model import TransformerModel

# Test 1: Overfitting on a small dataset
def test_overfitting():
    """Test if the model can overfit a small dataset."""
    vocab_size = 10
    embed_dim = 8
    num_heads = 2
    num_experts = 2
    expert_dim = 8

    model = TransformerModel(embed_dim, num_heads, num_experts, expert_dim, vocab_size)

    x = torch.randint(0, vocab_size, (2, 3))
    y = x.clone()
    dataset = TensorDataset(x, y)
    train_dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

    trainer = L.Trainer(max_epochs=200, accelerator="auto", devices=1, enable_progress_bar=False)
    trainer.fit(model, train_dataloader)

    with torch.no_grad():
        predictions = model.predict(x)
        accuracy = (predictions == y).float().mean().item()
        assert accuracy > 0.9, f"Model failed to overfit: accuracy = {accuracy}"
    print("Overfitting test passed!")

# Test 2: KV Cache consistency
def test_kv_cache():
    """Test if token-by-token generation with KV cache matches full sequence generation."""
    vocab_size = 10
    embed_dim = 8
    num_heads = 2
    num_experts = 2
    expert_dim = 8

    model = TransformerModel(embed_dim, num_heads, num_experts, expert_dim, vocab_size)

    input_ids = torch.randint(0, vocab_size, (1, 1))
    generated = [input_ids.item()]
    model.clear_cache()
    for _ in range(5):
        with torch.no_grad():
            output = model.predict(input_ids, use_cache=True)
            next_token = output[0, -1].item()
            generated.append(next_token)
            input_ids = torch.tensor([[next_token]])

    full_input = torch.tensor([generated[:-1]])
    with torch.no_grad():
        full_output = model.predict(full_input, use_cache=False)
        full_next_token = full_output[0, -1].item()

    assert full_next_token == generated[-1], "KV cache output differs from full sequence"
    print("KV cache test passed!")

# Test 3: Variable input sizes
def test_variable_sizes():
    """Test if the model handles different batch sizes and sequence lengths."""
    vocab_size = 10
    embed_dim = 8
    num_heads = 2
    num_experts = 2
    expert_dim = 8

    model = TransformerModel(embed_dim, num_heads, num_experts, expert_dim, vocab_size)

    inputs = [
        torch.randint(0, vocab_size, (1, 10)),
        torch.randint(0, vocab_size, (3, 5)),
        torch.randint(0, vocab_size, (2, 15)),
    ]
    expected_shapes = [(1, 10), (3, 5), (2, 15)]

    for x, expected_shape in zip(inputs, expected_shapes):
        with torch.no_grad():
            output = model.predict(x)
            assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    print("Variable sizes test passed!")

# Run all tests
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("Warning: GPU not available, tests may fail or run on CPU")
    
    test_overfitting()
    test_kv_cache()
    test_variable_sizes()
    print("All tests completed successfully!")