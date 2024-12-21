import os
import gc
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
import torch.multiprocessing as mp
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from model import RLLM
from dataset import ProteinRNADataset, collate_fn
from util import checkpoint, validate, load_config, plot, parse_data_file

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

def main():
    parser = argparse.ArgumentParser(description="Train or resume a model for RNA-Protein interaction.")
    parser.add_argument('--resume', type=str, default=None, help="Path to model checkpoint to resume training")
    parser.add_argument('--config', type=str, default="./config/config.yaml", help="Path to config file")
    parser.add_argument('--device', type=str, default="cuda", help="Device to use for training (cuda or cpu)")
    args = parser.parse_args()

    config = load_config(args.config)
    log_path = config["output_paths"]["log_path"]
    logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')

    checkpoint_dir = config["output_paths"]["checkpoint_dir"]
    plots_dir = config["output_paths"]["plots_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    device = torch.device(args.device if args.device else 'cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"])
    vocab_size = tokenizer.vocab_size

    model = RLLM(
        config["d_protein"], config["d_model"], 
        config["num_heads"], config["rllm_dropout"],
        config["gpt_dropout"], config["gpt_weights_path"], 
        vocab_size
    )

    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("RLLM model initialized successfully.")
    print(f"Number of parameters: {params/1e6:.2f}M")
    print(f"Number of trainable parameters: {trainable_params/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction="sum") 

    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        model_checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(model_checkpoint["model_state"])

        start_epoch = model_checkpoint["epoch"]
        iteration = model_checkpoint["iteration"]

        # Calculate start_index dynamically
        train_len = len(ProteinRNADataset(
            config["data_paths"]["pairs_train_path"],
            config["data_paths"]["protein_data_path"],
            tokenizer=tokenizer
        )) # total number of training samples
        gc.collect()

        start_index = (iteration * config["batch_size"]) % train_len
        
        print(f"Resuming training at epoch: {start_epoch} | iteration: {iteration} | start_index: {start_index}")
    else:
        start_epoch, iteration = 0, 0
        start_index = 0

    # DataLoader for Train and Validation
    train_dataset = ProteinRNADataset(
        config["data_paths"]["pairs_train_path"],
        config["data_paths"]["protein_data_path"],
        tokenizer=tokenizer
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        num_workers=config["num_workers_train"],
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        pin_memory=True
    )

    val_dataset = ProteinRNADataset(
        config["data_paths"]["pairs_val_path"],
        config["data_paths"]["protein_data_path"],
        tokenizer=tokenizer,
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        num_workers=config["num_workers_val"],
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        pin_memory=True
    )

    torch.set_float32_matmul_precision('high')
    model = torch.compile(model)
    model.to(device)

    num_epochs = config["num_epochs"]
    total_steps = len(train_dataloader) * num_epochs
    warmup_steps = int(config["warmup_ratio"] * total_steps) # learning rate warmup steps

    optimizer = optim.AdamW([
        {'params': model.get_trainable_parameters(cross_attn=True), 'lr': config["rllm_learning_rate"]},  # Cross-attention blocks
        {'params': model.get_trainable_parameters(gpt=True), 'lr': config["gpt_learning_rate"]},  # Trainable GPT layers
    ], weight_decay=config["optimizer_weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    if args.resume:
        optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        scheduler.load_state_dict(model_checkpoint["scheduler_state"])

    # Training loop
    training_losses = parse_data_file(config["output_paths"]["train_losses_path"])
    train_perplexities = parse_data_file(config["output_paths"]["train_perplexities_path"])
    validation_losses = parse_data_file(config["output_paths"]["val_losses_path"])
    validation_perplexities = parse_data_file(config["output_paths"]["val_perplexities_path"])
    train_losses_path, train_perplexities_path = config["output_paths"]["train_losses_path"], config["output_paths"]["train_perplexities_path"]
    val_losses_path, val_perplexities_path = config["output_paths"]["val_losses_path"], config["output_paths"]["val_perplexities_path"]
    
    early_stopping = config["early_stopping"]
    patience = config["early_stopping_patience"]  # Number of epochs with no improvement to wait

    best_val_loss = float('inf')
    val_loss_not_improved = 0

    checkpoint_interval = config["checkpoint_interval"]
    plot_interval = config["plot_interval"]

    start = time.time() # training start
    model.train()

    for epoch in range(start_epoch, num_epochs):
        offset = start_index if epoch == start_epoch else 0

        train_dataset = ProteinRNADataset(
            config["data_paths"]["pairs_train_path"],
            config["data_paths"]["protein_data_path"],
            tokenizer=tokenizer,
            offset=offset
        )
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=config["batch_size"], 
            num_workers=config["num_workers_train"],
            collate_fn=partial(collate_fn, tokenizer=tokenizer),
            pin_memory=True
        )

        # Calculate total batches for the current epoch
        total_batches = len(train_dataloader)
        
        # If resuming mid-epoch, set initial to the number of already processed batches
        if epoch == start_epoch and offset > 0:
            # Calculate how many batches have been processed in this epoch
            # Assuming start_index represents the starting batch index
            processed_batches = start_index // config["batch_size"]
            initial = processed_batches
            desc = f"Epoch {epoch + 1}/{num_epochs} (Resumed)"
        else:
            initial = 0
            desc = f"Epoch {epoch + 1}/{num_epochs}"

        running_train_loss, running_train_tokens = 0.0, 0

        with tqdm(train_dataloader, desc=desc, initial=initial, unit="batch", total=total_batches) as pbar:
            for batch_idx, batch in enumerate(pbar, start=initial):
                protein, protein_mask, rna_ids, rna_mask = (
                    batch["protein"].to(device), # embedding shape like [B, prot_len, d_protein]
                    batch["protein_mask"].to(device), # mask shape like [B, prot_len]
                    batch["rna"].to(device), # tokenized RNA sequence shape like [B, rna_len]
                    batch["rna_mask"].to(device) # mask shape like [B, rna_len]
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

                valid_tokens = (rna_tgt != tokenizer.pad_token_id).sum().item() # number of valid (non-padding) tokens
                
                running_train_loss += loss.item()
                running_train_tokens += valid_tokens

                if iteration % plot_interval == 0 and running_train_tokens > 0:
                    avg_train_loss = running_train_loss / running_train_tokens
                    avg_train_perplexity = torch.exp(torch.tensor(avg_train_loss)).item()

                    training_losses.append((avg_train_loss, iteration))
                    train_perplexities.append((avg_train_perplexity, iteration))

                    # Save training losses and perplexities
                    open(train_losses_path, 'a').write(f"{(avg_train_loss, iteration)}\n")
                    open(train_perplexities_path, 'a').write(f"{(avg_train_perplexity, iteration)}\n")

                    running_train_loss = 0.0 # reset
                    running_train_tokens = 0 # reset
                    
                    plot(training_losses, os.path.join(plots_dir, f"train_loss_epoch_{epoch}_iter_{iteration}.png"))
                    plot(train_perplexities, os.path.join(plots_dir, f"train_perplexity_epoch_{epoch}_iter_{iteration}.png"))

                if iteration % checkpoint_interval == 0:
                    avg_val_loss, avg_val_perplexity = validate(model, val_dataloader, criterion, device=device)
                    validation_losses.append((avg_val_loss, iteration))
                    validation_perplexities.append((avg_val_perplexity, iteration))

                    # Save validation losses and perplexities
                    open(val_losses_path, 'a').write(f"{(avg_val_loss, iteration)}\n")
                    open(val_perplexities_path, 'a').write(f"{(avg_val_perplexity, iteration)}\n")

                    plot(validation_losses, os.path.join(plots_dir, f"val_loss_epoch_{epoch}_iter_{iteration}.png"))
                    plot(validation_perplexities, os.path.join(plots_dir, f"val_perplexity_epoch_{epoch}_iter_{iteration}.png"))

                    checkpoint_filename = f"checkpoint_epoch_{epoch}_iter_{iteration}.pt"

                    if early_stopping and avg_val_loss >= best_val_loss:
                        val_loss_not_improved += 1
                        if val_loss_not_improved >= patience:
                            logging.info(f"Validation loss not improved for {patience} epochs. Early stopping triggered!")
                            break
                    if avg_val_loss < best_val_loss: # improvement in validation loss
                        best_val_loss = avg_val_loss
                        val_loss_not_improved = 0
                        checkpoint_filename = "best_" + checkpoint_filename
                        logging.info(f"Checkpointing best model with validation loss: {avg_val_loss:.6f}, at epoch {epoch + 1}, iteration {iteration}")
                    
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
                    checkpoint(model, optimizer, scheduler, epoch, iteration, checkpoint_path)

                # Update progress bar with current loss 
                num_valid_batch_tokens = (rna_tgt != tokenizer.pad_token_id).sum().item()
                pbar.set_postfix(loss=f"{loss.item() / num_valid_batch_tokens:.6f}", refresh=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_final.pt")
        checkpoint(model, optimizer, scheduler, (epoch + 1), iteration, checkpoint_path)

    end = time.time()
    print("Training complete!")
    print(f"Total training time: {end - start:.2f} seconds")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()