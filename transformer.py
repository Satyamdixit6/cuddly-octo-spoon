from encoder_feed import InputProcessor
from decoder_feed import OutputProcedssor

import torch.nn as nn 
import torch 
import torch.optim as optim
from torch.utils.data import DataLoader


class TransformerPipeline(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, num_heads_list, num_experts_list, expert_hidden_dim, max_seq_len=512):
        super(TransformerPipeline, self).__init__()
        self.input_processor = InputProcessor(vocab_size, embedding_dim, max_seq_len)
        self.output_processor = OutputProcessor(embedding_dim, vocab_size)
  

        #perameters for CUDA C++ implemetation
        self.num_layers = num_layers
        self.num_heads_list = num_heads_list
        self.num_experts_list = num_experts_list
        

    def forward(self,input_text):
        embedddings,input_ids = self.input_processor(input_text)
        cuda_output = transformer_cpp.transformer_model(
            input_ids,
            embeddings,
            self.num_heads_list,
            self.num_experts_list,
            self.expert_hidden_dim
        )

        logits = self.output_processor(cuda_output)

        return logits



import torch.optim as optim
from torch.utils.data import DataLoader

def train(model, dataloader, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_text, target_ids = batch  # Input text and target token IDs
            input_text, target_ids = input_text.cuda(), target_ids.cuda()

            # Forward pass
            logits = model(input_text)

            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader)}")









