from constants import RNA_LENGTH_CLUSTERS
from model import LengthPredictionHead

import os
from tqdm import tqdm
import logging
import time
import argparse
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

from dataset import ProteinRNALengthDataset, collate_fn_length
from util import checkpoint, validate_ln, load_config, plot, parse_data_file

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


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

    model = LengthPredictionHead(
        d_model=config["d_model"], num_classes=len(RNA_LENGTH_CLUSTERS),
        dropout=config["dropout"]
    )

    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Prediction head model initialized successfully.")
    print(f"Number of parameters: {params/1e6:.2f}M")
    print(f"Number of trainable parameters: {trainable_params/1e6:.2f}M")

    # create weights for cross-entropy loss to address class imbalance
    num_classes = len(RNA_LENGTH_CLUSTERS)
    counts = [0] * num_classes

    for cluster_id, _, count in RNA_LENGTH_CLUSTERS:
        counts[cluster_id] = count

    # Verify that all clusters are accounted for
    assert all(c > 0 for c in counts), "Some cluster counts are missing or zero."

    # Compute weights inversely proportional to counts
    total_counts = sum(counts)
    weights = [total_counts / (num_classes * c) for c in counts]

    weights = torch.tensor(weights, dtype=torch.float, device=device)

    # normalize the weights so that the minimum weight is 1
    min_weight = weights.min()
    weights = weights / min_weight
    criterion = FocalLoss(weight=weights)

    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        model_checkpoint = torch.load(args.resume, map_location=device)
        model_state_dict = model_checkpoint["model_state"]

        # pytorch compiled model state dict keys have "_orig_mod." prefix so we have to remove them
        for key in list(model_state_dict.keys()):
            model_state_dict[key.replace("_orig_mod.", "")] = model_state_dict.pop(key)
        model.load_state_dict(model_state_dict)

        start_epoch = model_checkpoint["epoch"]
        iteration = model_checkpoint["iteration"]
        
        print(f"Resuming training at epoch: {start_epoch} | iteration: {iteration}")
    else:
        start_epoch, iteration = 0, 0

    # DataLoader for Train and Validation
    train_dataset = ProteinRNALengthDataset(
        config["data_paths"]["pairs_train_path"],
        config["data_paths"]["protein_data_path"]
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        num_workers=config["num_workers_train"],
        collate_fn=collate_fn_length,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )

    val_dataset = ProteinRNALengthDataset(
        config["data_paths"]["pairs_val_path"],
        config["data_paths"]["protein_data_path"]
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        num_workers=config["num_workers_val"],
        collate_fn=collate_fn_length,
        pin_memory=True,
        shuffle=False,
        drop_last=True
    )

    torch.set_float32_matmul_precision('high')
    model = torch.compile(model) 
    model.to(device)

    num_epochs = config["num_epochs"]

    optimizer = optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["optimizer_weight_decay"]
    )
    """ scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config["scheduler"]["factor"], patience=config["scheduler"]["patience"],
        threshold=config["scheduler"]["threshold"], threshold_mode='abs', min_lr=config["scheduler"]["min_lr"]
    ) """
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        steps_per_epoch=len(train_dataloader),
        epochs=num_epochs,
        pct_start=0.3  # Spend 30% of training warming up
    )

    if args.resume:
        optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        scheduler.load_state_dict(model_checkpoint["scheduler_state"])

    # Training loop
    train_losses_path = config["output_paths"]["train_losses_path"]
    val_losses_path = config["output_paths"]["val_losses_path"]
    training_losses = parse_data_file(train_losses_path)
    validation_losses = parse_data_file(val_losses_path)
    
    early_stopping = config["early_stopping"]
    patience = config["early_stopping_patience"]  # Number of epochs with no improvement to wait

    best_val_loss = float("inf") if not args.resume else min([loss for loss, _ in validation_losses])
    val_loss_not_improved = 0

    checkpoint_interval = config["checkpoint_interval"]
    plot_interval = config["plot_interval"]
    validation_interval = config["validation_interval"]

    start = time.time() # training start
    model.train()

    for epoch in range(start_epoch, num_epochs):

        running_train_loss = 0.0

        with tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", total=len(train_dataloader)) as pbar:
            for batch_idx, batch in enumerate(pbar):
                protein, protein_mask, labels = (
                    batch["protein"].to(device), # embedding shape like [B, prot_len, d_protein]
                    batch["protein_mask"].to(device), # mask shape like [B, prot_len]
                    batch["labels"].to(device), # shape like [B]
                )
                optimizer.zero_grad()

                logits = model(protein, protein_mask) # shape like [B, num_classes]
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

                iteration += 1

                running_train_loss += loss.item()

                if iteration % plot_interval == 0:
                    avg_train_loss = running_train_loss / plot_interval

                    training_losses.append((avg_train_loss, iteration))

                    # Save training losses and perplexities
                    open(train_losses_path, 'a').write(f"{(avg_train_loss, iteration)}\n")
                    plot(training_losses, os.path.join(plots_dir, f"train_loss_epoch_{epoch}_iter_{iteration}.png"))

                    running_train_loss = 0.0 # reset

                if iteration % validation_interval == 0:
                    avg_val_loss = validate_ln(model, val_dataloader, criterion, device=device)
                    validation_losses.append((avg_val_loss, iteration))

                    # Save validation losses and perplexities
                    open(val_losses_path, 'a').write(f"{(avg_val_loss, iteration)}\n")
                    plot(validation_losses, os.path.join(plots_dir, f"val_loss_epoch_{epoch}_iter_{iteration}.png"))

                    if avg_val_loss < best_val_loss: # improvement in validation loss
                        best_val_loss = avg_val_loss
                        val_loss_not_improved = 0

                        checkpoint_filename = f"best_checkpoint_epoch_{epoch}_iter_{iteration}.pt"
                        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
                        logging.info(f"Checkpointing best model with validation loss: {avg_val_loss:.6f}, at epoch {epoch + 1}, iteration {iteration}")
                        checkpoint(model, optimizer, scheduler, epoch, iteration, checkpoint_path)

                        if early_stopping and avg_val_loss >= best_val_loss:
                            val_loss_not_improved += 1
                            if val_loss_not_improved >= patience:
                                break

                if iteration % checkpoint_interval == 0:
                    checkpoint_filename = f"checkpoint_epoch_{epoch}_iter_{iteration}.pt"
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
                    checkpoint(model, optimizer, scheduler, epoch, iteration, checkpoint_path)

                    # Iterate through each parameter group and log its name and learning rate
                    for group in optimizer.param_groups:
                        lr = group.get('lr', None)
                        logging.info(f"Learning rate: {lr}")

                # Update progress bar with current loss 
                pbar.set_postfix(loss=f"{loss.item():.6f}", refresh=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_final.pt")
        checkpoint(model, optimizer, scheduler, epoch, iteration, checkpoint_path)

        if early_stopping and val_loss_not_improved >= patience:
            break

    end = time.time()
    if early_stopping and val_loss_not_improved >= patience:
        print(f"Early stopping triggered! Training stopped at epoch {epoch + 1}")
    else:
        print("Training complete!")
    print(f"Total training time: {end - start:.2f} seconds")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()