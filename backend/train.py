import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import re
from model import MiniGPT
import torch.nn.functional as F

# Your Google Colab Dataset
sentences = [
    "I love AI",
    "I love coding",
    "AI is powerful",
    "Transformers are amazing",
    "I explore AI"
]
text = " . ".join(sentences) + " ."

def tokenize(text):
    # Simple word tokenizer: split by non-alphanumeric, keeping words (case-sensitive)
    words = re.findall(r'\b\w+\b|[^\w\s]', text)
    return words

def train():
    print("Preparing data...")
    words = tokenize(text)
    
    # Create vocabulary mapping
    unique_words = sorted(list(set(words)))
    
    # Add special tokens
    special_tokens = ["<PAD>", "<UNK>"]
    vocab = special_tokens + unique_words
    
    vocab_size = len(vocab)
    stoi = { w:i for i,w in enumerate(vocab) }
    itos = { i:w for i,w in enumerate(vocab) }

    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    with open("vocab.json", "w") as f:
        json.dump({"stoi": stoi, "itos": itos}, f)

    data = torch.tensor([stoi[w] for w in words], dtype=torch.long)
    
    block_size = 8 # Context length (max_length for words)
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MiniGPT(
        vocab_size=vocab_size,
        max_length=block_size,
        embed_size=32,
        num_layers=1,
        heads=2,
        forward_expansion=4,
        dropout=0.1
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # We must define loss_function to match your loop
    loss_function = nn.CrossEntropyLoss()

    # Prepare inputs and targets (Full Batch) to match your Colab logic
    inputs_list = []
    targets_list = []
    for i in range(len(data) - block_size):
        inputs_list.append(data[i:i+block_size])
        targets_list.append(data[i+1:i+block_size+1])
    
    if len(inputs_list) == 0:
        padded_data = F.pad(data, (0, block_size - len(data)), value=stoi["<PAD>"])
        inputs_list.append(padded_data)
        targets_list.append(F.pad(data[1:], (0, block_size - len(data) + 1), value=stoi["<PAD>"]))
        
    inputs = torch.stack(inputs_list).to(device)
    targets = torch.stack(targets_list).to(device)

    print(f"Training model on {device}...")
    
    # === YOUR GOOGLE COLAB TRAINING LOOP ===
    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # PyTorch requires us to reshape the outputs for the loss function
        raw_outputs = model(inputs)
        B, T, C = raw_outputs.shape
        outputs = raw_outputs.view(B * T, C)
        targets_reshaped = targets.view(B * T)
        
        loss = loss_function(outputs, targets_reshaped)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
    # =======================================

    print("Training complete. Saving model...")
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

if __name__ == "__main__":
    train()