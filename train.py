import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial

from model import ParallelizedCrossAttentionModel
from dataset import ProteinRNADataset, collate_fn
from tokenizer import RNATokenizer

# Hyperparameters
batch_size = 2
num_layers = 12
d_rna = 768
d_protein = 1536
d_model = 768
num_heads = 8
vocab_size = 8
num_epochs = 10
learning_rate = 1e-4

protein_data_path = "/home/bil/dataset/embeddings/protein"
rna_data_path = "/home/bil/dataset/embeddings/rna"
pairs_data_path = "/home/bil/dataset/embeddings/pairs.txt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tokenizer = RNATokenizer()

# DataLoader
dataset = ProteinRNADataset(
    pairs_file=pairs_data_path,
    protein_folder=protein_data_path, 
    rna_folder=rna_data_path,
    tokenizer=tokenizer
)
dataloader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    collate_fn=partial(collate_fn, tokenizer=tokenizer)
)

# Model, Loss, Optimizer
model = ParallelizedCrossAttentionModel(
    d_rna=d_rna, d_protein=d_protein, d_model=d_model,
    num_heads=num_heads, num_layers=num_layers, vocab_size=vocab_size
).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token) 

# Training Loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for sample in dataloader:
        protein_batch, rna_batch, protein_mask, rna_mask, target_batch = (
            sample["protein"].to(device), 
            sample["rna"].to(device), 
            sample["protein_mask"].to(device), 
            sample["rna_mask"].to(device), 
            sample["rna_targets"].to(device)
        )
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(
            rna_batch,
            protein_batch,
            rna_mask,
            protein_mask
        ) # [B, seq_len, vocab_size]
        
        # Reshape for loss computation
        logits = logits.view(-1, vocab_size) # [B * seq_len, vocab_size]
        targets = target_batch.view(-1)      # [B * seq_len]
        
        # Compute loss
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        print(loss.item())
    
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")

print("Training complete!")