# DeepSeek-V3

A PyTorch implementation of the DeepSeek-V3 language model architecture, featuring Multi-head Latent Attention (MLA), Mixture of Experts (MoE), and Multi-Token Prediction (MTP).

## Features

- Multi-head Latent Attention (MLA) for efficient attention computation
- Mixture of Experts (MoE) for improved model capacity
- Multi-Token Prediction (MTP) for enhanced training efficiency
- Configurable architecture parameters
- PyTorch-based implementation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deepseek-v3.git
cd deepseek-v3
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
deepseek_v3/
├── README.md
├── requirements.txt
└── src/
    ├── config/
    │   └── model_config.py
    ├── model/
    │   ├── mla.py
    │   ├── moe.py
    │   ├── mtp.py
    │   └── model.py
    └── example_usage.py
```

## Usage

Here's a basic example of how to use the model:

```python
import torch
from config.model_config import DeepSeekConfig
from model.model import DeepSeekV3

# Create model configuration
config = DeepSeekConfig(
    vocab_size=32000,
    hidden_size=2048,
    num_hidden_layers=24,
    num_attention_heads=32,
    latent_dim=128,
    num_experts=8,
    num_selected_experts=2,
    expert_hidden_size=4096,
    shared_expert_size=1024,
    num_mtp_layers=4,
    max_position_embeddings=4096,
    dropout=0.1,
    attention_dropout=0.1,
    activation_function="gelu",
    load_balancing_weight=0.01,
    bias_update_speed=0.01,
    mtp_weight=0.1
)

# Initialize model
model = DeepSeekV3(config)

# Create example input
batch_size = 2
seq_length = 32
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
attention_mask = torch.ones((batch_size, seq_length))

# Forward pass
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_attentions=True,
    output_hidden_states=True
)
```

To run the example script:
```bash
python src/example_usage.py
```

## Model Architecture

### Multi-head Latent Attention (MLA)
- Efficient attention computation using latent vectors
- Reduced memory complexity
- Improved computational efficiency

### Mixture of Experts (MoE)
- Multiple expert networks
- Dynamic routing mechanism
- Load balancing for expert utilization

### Multi-Token Prediction (MTP)
- Parallel prediction of multiple tokens
- Enhanced training efficiency
- Improved model performance

## Configuration

The model can be configured using the `DeepSeekConfig` class. Key parameters include:

- `vocab_size`: Size of the vocabulary
- `hidden_size`: Dimension of the hidden states
- `num_hidden_layers`: Number of transformer layers
- `num_attention_heads`: Number of attention heads
- `latent_dim`: Dimension of latent vectors in MLA
- `num_experts`: Number of experts in MoE
- `num_selected_experts`: Number of experts to use per token
- `expert_hidden_size`: Hidden size of expert networks
- `shared_expert_size`: Size of shared expert network
- `num_mtp_layers`: Number of MTP layers
- `max_position_embeddings`: Maximum sequence length
- `dropout`: Dropout probability
- `attention_dropout`: Attention dropout probability
- `activation_function`: Activation function for experts
- `load_balancing_weight`: Weight for load balancing loss
- `bias_update_speed`: Speed of expert bias updates
- `mtp_weight`: Weight for MTP loss

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 