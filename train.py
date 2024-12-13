import os
from functools import partial
from tqdm import tqdm
import logging
import time
import argparse
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import ParallelizedCrossAttentionModel
from dataset import ProteinRNADataset, collate_fn
from tokenizer import RNATokenizer
from util import checkpoint, validate, load_config, get_optimizer, get_scheduler, plot_loss

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

parser = argparse.ArgumentParser(description="Train or resume a model for RNA-Protein interaction.")
parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint file to resume training")
parser.add_argument('--config', type=str, default="./config/config.yaml", help="Path to config file")
args = parser.parse_args()

config = load_config(args.config)
logging.info("Configuration file loaded successfully.")

log_path = config["log_path"]
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')

checkpoint_dir = config["directories"]["checkpoint_dir"]
plots_dir = config["directories"]["plots_dir"]
epochs_plots_dir = config["directories"]["epochs_plots_dir"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(epochs_plots_dir, exist_ok=True)


# Initialize tokenizer
tokenizer = RNATokenizer()
vocab_size = tokenizer.vocab_size

# DataLoader for Train and Validation
train_dataset = ProteinRNADataset(
    pairs_file=config["data_paths"]["pairs_train_path"],
    protein_folder=config["data_paths"]["protein_data_path"], 
    rna_folder=config["data_paths"]["rna_data_path"],
    tokenizer=tokenizer
)
val_dataset = ProteinRNADataset(
    pairs_file=config["data_paths"]["pairs_val_path"],
    protein_folder=config["data_paths"]["protein_data_path"], 
    rna_folder=config["data_paths"]["rna_data_path"],
    tokenizer=tokenizer
)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=config["batch_size"], 
    collate_fn=partial(collate_fn, tokenizer=tokenizer)
)
val_dataloader = DataLoader(
    val_dataset, 
    batch_size=config["batch_size"], 
    collate_fn=partial(collate_fn, tokenizer=tokenizer)
)


# Model, Loss, Optimizer, Scheduler
model = ParallelizedCrossAttentionModel(
    d_rna=config["d_rna"], d_protein=config["d_protein"], d_model=config["d_model"],
    num_heads=config["num_heads"], num_layers=config["num_layers"], vocab_size=vocab_size
).to(device)

trainable_params = sum(p.numel() for p in model.parameters())
print(f"Number of Trainable Parameters: {trainable_params}")

optimizer = get_optimizer(model, config["optimizer"], config["learning_rate"])
scheduler = get_scheduler(optimizer, config["scheduler"])
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token) 

start_epoch, iteration = 0, 0
epoch_train_losses, epoch_val_losses = [], []
training_losses, validation_losses = [], []
best_val_loss = float('inf')


# Training Loop
epoch_train_losses, epoch_val_losses = [], []
training_losses, validation_losses = [], []

num_epochs = config["num_epochs"]

iteration = 0

# Resume training if specified
if args.resume:
    print(f"Resuming training from checkpoint: {args.resume}")
    checkpoint_data = torch.load(args.resume, map_location=device)

    model.load_state_dict(checkpoint_data["model_state"])
    optimizer.load_state_dict(checkpoint_data["optimizer_state"])
    scheduler.load_state_dict(checkpoint_data["scheduler_state"])

    start_epoch = checkpoint_data["epoch"]
    iteration = checkpoint_data["iteration"]

    print(f"Resumed at Epoch {start_epoch}, Iteration {iteration}")

early_stopping_patience = config["patience"]
epochs_without_improvement = 0
checkpoint_interval = config["checkpoint_interval"]
plot_interval = config["plot_interval"]
best_val_loss = float('inf')

start = time.time()

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_train_loss = 0.0
    running_loss = 0.0

    current_iteration = 0

    with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for batch_idx, sample in enumerate(pbar):
            current_iteration += 1

            if current_iteration <= (iteration % len(train_dataloader)):
                continue # Skip if resuming from a checkpoint

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
            total_train_loss += loss.item()
            running_loss += loss.item()

            if iteration % plot_interval == 0:
                training_losses.append(running_loss / plot_interval)
                running_loss = 0.0

                avg_val_loss = validate(model, val_dataloader, criterion, vocab_size, device, subset_size=10)
                validation_losses.append(avg_val_loss)
                scheduler.step(avg_val_loss)

                plot_loss(training_losses, validation_losses, os.path.join(plots_dir, f"epoch_{epoch}_iter_{iteration}_intermediate.png"))

            if iteration % checkpoint_interval == 0:
                logging.info(f"Checkpointing: Epoch {epoch}, Iteration {iteration}, Loss: {loss.item():.4f}")
                checkpoint(
                    model, optimizer, scheduler, epoch, iteration, 
                    checkpoint_dir, plot=False
                )
                
            # Update progress bar with current loss
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Average Training Loss
    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_val_loss = validate(model, val_dataloader, criterion, vocab_size, device)
    epoch_train_losses.append(avg_train_loss)
    epoch_val_losses.append(avg_val_loss)

    checkpoint(
        model, optimizer, scheduler, epoch, iteration, 
        checkpoint_dir, epochs_plots_dir, epoch_train_losses, epoch_val_losses,
        plot=True
    )

    if avg_val_loss - best_val_loss < -(1e-5): # Slight improvement
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        
        logging.info(f"Best model with validation loss: {avg_val_loss:.4f}, at epoch {epoch + 1}, iteration {iteration}")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= early_stopping_patience:
        print("Early stopping triggered.")
        break

    logging.info(f"Turning an epoch. Current status => Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    logging.info(f"Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}\n")

print("Training complete!")

if not args.resume:
    print(f"Trained {epoch + 1} epochs in {time.time() - start:.2f} seconds.")
