import os
from functools import partial
from tqdm import tqdm
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ParallelizedCrossAttentionModel
from dataset import ProteinRNADataset, collate_fn
from tokenizer import RNATokenizer
from util import checkpoint, validate, plot_epoch_losses

# Hyperparameters
batch_size = 2
num_layers = 12
d_rna = 768
d_protein = 1536
d_model = 768
num_heads = 8
num_epochs = 10
learning_rate = 1e-4
checkpoint_interval = 500

protein_data_path = "./dataset/protein"
rna_data_path = "./dataset/rna"

pairs_train_path = "./dataset/train.txt"
pairs_val_path = "./dataset/val.txt"
pairs_test_path = "./dataset/test.txt"

log_path = "./train.log"
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')

checkpoint_dir = "./checkpoints"
plots_dir = "./plots"
epochs_plots_dir = os.path.join(plots_dir, "epochs")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)


# Initialize tokenizer
tokenizer = RNATokenizer()
vocab_size = tokenizer.vocab_size

# DataLoader for Train and Validation
train_dataset = ProteinRNADataset(
    pairs_file=pairs_train_path,
    protein_folder=protein_data_path, 
    rna_folder=rna_data_path,
    tokenizer=tokenizer
)
val_dataset = ProteinRNADataset(
    pairs_file=pairs_val_path,
    protein_folder=protein_data_path, 
    rna_folder=rna_data_path,
    tokenizer=tokenizer
)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    collate_fn=partial(collate_fn, tokenizer=tokenizer)
)
val_dataloader = DataLoader(
    val_dataset, 
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


train_losses, val_losses = [], []
epoch_train_losses, epoch_val_losses = [], []

# Training Loop
start = time.time()
iteration = 0
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0
    total_train_loss = 0.0

    # Progress bar for current epoch
    with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for batch_idx, sample in enumerate(pbar):
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
            
            iteration += 1
            running_train_loss += loss.item()
            total_train_loss += loss.item()

            # Save intermediate checkpoint and validation loss
            if iteration % checkpoint_interval == 0:
                avg_train_loss = running_train_loss / checkpoint_interval
                running_train_loss = 0.0  # Reset running loss

                val_loss = validate(model, val_dataloader, criterion, vocab_size, device, subset_size=50)
                train_losses.append(avg_train_loss)
                val_losses.append(val_loss)

                logging.info(f"\nIteration {iteration}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                checkpoint(model, optimizer, epoch, iteration, train_losses, val_losses, checkpoint_dir, plots_dir)

            # Update progress bar with current loss
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Average Training Loss
    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_val_loss = validate(model, val_dataloader, criterion, vocab_size, device)
    epoch_train_losses.append(avg_train_loss)
    epoch_val_losses.append(avg_val_loss)

    logging.info(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save final checkpoint and plot after epoch
    checkpoint(model, optimizer, epoch, iteration, avg_train_loss, train_losses, val_losses, checkpoint_dir, plots_dir)
    plot_epoch_losses(train_losses, val_losses, epoch, epochs_plots_dir)

print("Training complete!")
print(f"Trained {num_epochs} epochs in {time.time() - start:.2f} seconds.")