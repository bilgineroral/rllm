import os
import yaml
import logging
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, OneCycleLR

from model import ParallelizedCrossAttentionModel

def plot_loss(train_losses: list, val_losses: list, plot_path: str) -> None:
    """
    Plots and saves the training and validation loss.

    Args:
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        plot_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
    plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(plot_path)
    plt.close()

# Checkpoint Function
def checkpoint(model: ParallelizedCrossAttentionModel,
               optimizer: torch.optim.Optimizer,
               scheduler,
               epoch: int,
               iteration: int, 
               checkpoint_dir: str,
               plots_dir: str = None,
               train_losses: list = None,
               val_losses: list = None,
               plot=False
            ) -> None:
    """
    Saves a model checkpoint and the training/validation loss plot.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        iteration (int): Current training iteration.
        epoch (int): Current epoch.
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
    """

    checkpoint = {
        "epoch": epoch,
        "iteration": iteration,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_iter_{iteration}.pt")
    torch.save(checkpoint, checkpoint_path)

    if plot:
        if train_losses is None or val_losses is None or plots_dir is None:
            raise ValueError("train_losses, val_losses, and plots_dir must be provided to plot losses.")
        # Plot and save training/validation loss
        plot_path = os.path.join(plots_dir, f"loss_plot_epoch_{epoch}_iter_{iteration}.png")
        plot_loss(train_losses, val_losses, plot_path)


def validate(model: ParallelizedCrossAttentionModel, 
             dataloader: torch.utils.data.DataLoader, 
             criterion: torch.nn.Module,
             vocab_size: int, 
             device: torch.device, 
             subset_size: int = None
            ) -> float:
    """
    Validates the model on the validation set or a subset of it.
    
    Args:
        model (torch.nn.Module): The model to validate.
        dataloader (torch.utils.data.DataLoader): The validation DataLoader.
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss).
        vocab_size (int): Vocabulary size for the output layer.
        device (torch.device): Device to run validation on.
        subset_size (int, optional): Number of batches to use for subset validation. 
                                     If None, validates on the entire dataset.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        # Use subset if subset_size is provided
        dataloader_iter = iter(dataloader)
        batches_to_evaluate = subset_size if subset_size is not None else len(dataloader)

        for _ in range(batches_to_evaluate):
            try:
                sample = next(dataloader_iter)
            except StopIteration:
                break  # End early if dataset is smaller than subset_size

            protein_batch, rna_batch, protein_mask, rna_mask, target_batch = (
                sample["protein"].to(device), 
                sample["rna"].to(device), 
                sample["protein_mask"].to(device), 
                sample["rna_mask"].to(device), 
                sample["rna_targets"].to(device)
            )
            logits = model(rna_batch, protein_batch, rna_mask, protein_mask)
            logits = logits.view(-1, vocab_size)
            targets = target_batch.view(-1)
            total_loss += criterion(logits, targets).item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float('inf')


def load_config(config_path: str):
    """
    Load YAML configuration file and set up logging.

    Args:
        config_path (str): Path to the config.yaml file.
    
    Returns:
        dict: Configuration dictionary with all settings.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Logging setup
    log_path = config.get("log_path", "./train.log")
    logging.basicConfig(
        filename=log_path, 
        level=logging.INFO, 
        format='%(asctime)s - %(message)s'
    )
    logging.info("Configuration file loaded successfully.")
    return config

def get_optimizer(model, optimizer_config, lr):
    """Initialize optimizer based on configuration."""
    opt_type = optimizer_config["type"]
    weight_decay = optimizer_config.get("weight_decay", 0.0)

    if opt_type == "AdamW":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "SGD":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

def get_scheduler(optimizer, scheduler_config):
    """Initialize scheduler based on configuration."""
    sched_type = scheduler_config["type"]
    params = scheduler_config.get("params", {})

    assert sched_type == "ReduceLROnPlateau", "Only ReduceLROnPlateau is supported for now."

    if sched_type == "ReduceLROnPlateau":
        return ReduceLROnPlateau(optimizer, **params)
    elif sched_type == "StepLR":
        return StepLR(optimizer, step_size=params.get("step_size", 10), gamma=params.get("gamma", 0.1))
    elif sched_type == "CosineAnnealingLR":
        return CosineAnnealingLR(optimizer, T_max=params.get("T_max", 10), eta_min=params.get("eta_min", 1e-6))
    elif sched_type == "OneCycleLR":
        return OneCycleLR(optimizer, max_lr=params.get("max_lr", 1e-3), total_steps=params.get("total_steps", 1000))
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")
