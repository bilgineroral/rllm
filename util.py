import os
import matplotlib.pyplot as plt

import torch

from model import ParallelizedCrossAttentionModel

def plot_epoch_losses(epoch_train_losses: list, 
                epoch_val_losses: list, 
                epoch: int, 
                plots_dir: str):
    """
    Plots the training and validation losses over epochs.

    Args:
        epoch_train_losses (list): List of average training losses per epoch.
        epoch_val_losses (list): List of average validation losses per epoch.
        plots_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_train_losses, label='Training Loss')
    plt.plot(epoch_val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plots_dir}/epoch_{epoch}_loss_plot.png")
    plt.close()


# Checkpoint Function
def checkpoint(model: ParallelizedCrossAttentionModel,
               optimizer: torch.optim.Optimizer,
               epoch: int,
               iteration: int, 
               train_losses: list,
               val_losses: list,
               checkpoint_dir: str,
               plots_dir: str
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
    }
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_iter_{iteration}.pt")
    torch.save(checkpoint, checkpoint_path)

    # Plot and save training/validation loss
    plot_path = os.path.join(plots_dir, f"loss_plot_epoch_{epoch}_iter_{iteration}.png")
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
