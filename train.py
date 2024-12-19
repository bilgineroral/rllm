import math
import os
from functools import partial
from tqdm import tqdm
import logging
import time
import argparse
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from model import RLLM
from dataset import ProteinRNADataset, collate_fn
from util import checkpoint, validate, load_config, plot

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

parser = argparse.ArgumentParser(description="Train or resume a model for RNA-Protein interaction.")
parser.add_argument('--resume', type=str, default=None, help="Path to model checkpoint to resume training")
parser.add_argument('--config', type=str, default="./config/config.yaml", help="Path to config file")
args = parser.parse_args()

config = load_config(args.config)

log_path = config["log_path"]
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("Configuration file loaded successfully.")

checkpoint_dir = config["directories"]["checkpoint_dir"]
plots_dir = config["directories"]["plots_dir"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
vocab_size = tokenizer.vocab_size

# DataLoader for Train and Validation
train_dataset = ProteinRNADataset(
    config["data_paths"]["pairs_train_path"],
    config["data_paths"]["protein_data_path"],
    tokenizer=tokenizer,
    device=device
)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=config["batch_size"], 
    collate_fn=partial(collate_fn, tokenizer=tokenizer, device=device)
)

val_dataset = ProteinRNADataset(
    config["data_paths"]["pairs_val_path"],
    config["data_paths"]["protein_data_path"],
    tokenizer=tokenizer,
    device=device
)
val_dataloader = DataLoader(
    val_dataset, 
    batch_size=config["batch_size"], 
    collate_fn=partial(collate_fn, tokenizer=tokenizer, device=device)
)

# Model, Loss, Optimizer, Scheduler
model = RLLM(
    config["d_protein"], config["d_model"], 
    config["num_heads"], config["rllm_dropout"],
    config["gpt_dropout"], config["gpt_weights_path"], 
    vocab_size
).to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("RLLM model initialized successfully.")
print(f"Number of trainable parameters: {trainable_params/1e6:.2f}M")

num_epochs = config["num_epochs"]

total_steps = len(train_dataloader) * num_epochs
warmup_steps = int(config["warmup_ratio"] * total_steps) # learning rate warmup steps

optimizer = optim.AdamW([
    {'params': model.get_trainable_parameters(cross_attn=True), 'lr': config["rllm_learning_rate"]},  # Cross-attention blocks
    {'params': model.get_trainable_parameters(layer_norm=True), 'lr': config["rllm_learning_rate"]},  # LayerNorm blocks
    {'params': model.get_trainable_parameters(gpt=True), 'lr': config["gpt_learning_rate"]},  # Trainable GPT layers
], weight_decay=config["optimizer_weight_decay"])
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="sum") 

# Training loop
start_epoch, iteration = 0, 0
training_losses, train_perplexities = [], [] # list of train. losses and perplexity scores saved at a fixed interval of steps
validation_losses, validation_perplexities = [], [] # list of val. losses and perplexity scores saved at a fixed interval of steps
best_val_loss = float('inf') # best validation loss throughout the entire training

if args.resume:
    print(f"Resuming training from checkpoint: {args.resume}")
    model_checkpoint = torch.load(args.resume, map_location=device)

    model.load_state_dict(model_checkpoint["model_state"])
    optimizer.load_state_dict(model_checkpoint["optimizer_state"])
    scheduler.load_state_dict(model_checkpoint["scheduler_state"])

    start_epoch = model_checkpoint["epoch"]
    iteration = model_checkpoint["iteration"]

    print(f"Resuming training at epoch: {start_epoch} | iteration: {iteration}")

checkpoint_interval = config["checkpoint_interval"]
plot_interval = config["plot_interval"]

start = time.time() # training start

for epoch in range(start_epoch, num_epochs):
    model.train()

    running_train_loss = 0.0
    running_train_tokens = 0
    current_iteration = 0

    with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
        for batch in pbar:
            current_iteration += 1

            if args.resume and current_iteration < iteration % len(train_dataloader):
                continue

            protein, protein_mask, rna_ids, rna_mask = (
                batch["protein"], # embedding shape like [B, prot_len, d_protein]
                batch["protein_mask"], # mask shape like [B, prot_len]
                batch["rna"], # tokenized RNA sequence shape like [B, rna_len]
                batch["rna_mask"] # mask shape like [B, rna_len]
            )
            optimizer.zero_grad()

            rna_src = rna_ids[:, :-1] # remove last token for source 
            rna_mask = rna_mask[:, :-1] # remove last token for source
            rna_tgt = rna_ids[:, 1:] # remove first token for target

            logits = model(protein, rna_src, protein_mask, rna_mask)
            logits = logits.reshape(-1, vocab_size)
            rna_tgt = rna_tgt.reshape(-1)

            loss = criterion(logits, rna_tgt)
            loss.backward()
            
            optimizer.step()
            scheduler.step()

            iteration += 1

            pad_token_id = tokenizer.pad_token_id
            valid_tokens = (rna_tgt != pad_token_id).sum().item() # number of valid (non-padding) tokens
            
            running_train_loss += loss.item()
            running_train_tokens += valid_tokens

            if iteration % plot_interval == 0 and running_train_tokens > 0:
                avg_train_loss = running_train_loss / running_train_tokens
                avg_train_perplexity = math.exp(avg_train_loss)

                training_losses.append((avg_train_loss, iteration))
                train_perplexities.append((avg_train_perplexity, iteration))

                running_train_loss = 0.0 # reset
                running_train_tokens = 0 # reset
                
                plot(training_losses, os.path.join(plots_dir, f"train_loss_epoch_{epoch}_iter_{iteration}.png"))
                plot(train_perplexities, os.path.join(plots_dir, f"train_perplexity_epoch_{epoch}_iter_{iteration}.png"))

            if iteration % checkpoint_interval == 0:
                avg_val_loss, avg_val_perplexity = validate(model, val_dataloader, criterion)
                validation_losses.append((avg_val_loss, iteration))
                validation_perplexities.append((avg_val_perplexity, iteration))

                plot(validation_losses, os.path.join(plots_dir, f"val_loss_epoch_{epoch}_iter_{iteration}.png"))
                plot(validation_perplexities, os.path.join(plots_dir, f"val_perplexity_epoch_{epoch}_iter_{iteration}.png"))

                checkpoint_filename = f"checkpoint_epoch_{epoch}_iter_{iteration}.pt"

                if avg_val_loss < best_val_loss: # improvement in validation loss
                    best_val_loss = avg_val_loss
                    checkpoint_filename = "best_" + checkpoint_filename
                    logging.info(f"Checkpointing best model with validation loss: {avg_val_loss:.6f}, at epoch {epoch + 1}, iteration {iteration}")
                
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
                checkpoint(model, optimizer, scheduler, epoch, iteration, checkpoint_path)

            # Update progress bar with current loss 
            num_valid_batch_tokens = (rna_tgt != pad_token_id).sum().item()
            pbar.set_postfix(loss=f"{loss.item() / num_valid_batch_tokens:.6f}", refresh=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_final.pt")
    checkpoint(model, optimizer, scheduler, epoch, iteration, checkpoint_path)

end = time.time()
print("Training complete!")
print(f"Total training time: {end - start:.2f} seconds")
