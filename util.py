import os
import yaml
import logging
import matplotlib.pyplot as plt

import torch

from model import RLLM

def plot(data: list, save_path: str) -> None:
    """
    Plots loss/perplexity values against iterations and saves the plot to a specified path.
    
    Args:
        data (list of tuples): A list where each tuple contains (value, iteration).
        save_path (str): The file path to save the generated plot.
    """
    # Unpack loss values and iterations
    loss_values, iterations = zip(*data)
    
    # Plot the loss values
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, loss_values, marker='o', linestyle='-', linewidth=2)
    plt.xticks(iterations) 
    plt.xlabel('Iterations')
    plt.ylabel('Loss Value')
    plt.title('Loss vs Iterations')
    plt.grid(True)
    
    # Save the plot
    plt.savefig(save_path)
    plt.close()


def checkpoint(model: RLLM,
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler.LambdaLR,
               epoch: int,
               iteration: int, 
               checkpoint_path: str,
            ) -> None:
    """
    Saves a model checkpoint and the training/validation loss plot.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): Optimizer to save.
        scheduler (torch.optim.lr_scheduler.LambdaLR): Scheduler to save.
        epoch (int): Current epoch.
        iteration (int): Current training iteration.
        checkpoint_path (str): Path to save the checkpoint.
    """

    checkpoint = {
        "epoch": epoch,
        "iteration": iteration,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)


def validate(model: RLLM, 
             dataloader: torch.utils.data.DataLoader, 
             criterion: torch.nn.Module,
            ) -> float:
    """
    Validates the model on the validation set or a subset of it.
    
    Args:
        model (torch.nn.Module): The model to validate.
        dataloader (torch.utils.data.DataLoader): The validation DataLoader.
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss).

    Returns:
        float: Perplexity score.
    """
    assert type(criterion) == torch.nn.CrossEntropyLoss, "Criterion must be CrossEntropyLoss"
    assert criterion.reduction == "sum", "Criterion must have reduction='sum' for validation"

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    pad_token_id = criterion.ignore_index

    with torch.no_grad():
        for batch in dataloader:
            protein, protein_mask, rna_ids, rna_mask = (
                batch["protein"], # embedding shape like [B, prot_len, d_protein]
                batch["protein_mask"], # mask shape like [B, prot_len]
                batch["rna"], # tokenized RNA sequence shape like [B, rna_len]
                batch["rna_mask"] # mask shape like [B, rna_len]
            )

            rna_src = rna_ids[:, :-1] # remove last token for source
            rna_mask = rna_mask[:, :-1]
            rna_tgt = rna_ids[:, 1:]

            logits = model(protein, rna_src, protein_mask, rna_mask)
            logits = logits.reshape(-1, logits.shape[-1])
            rna_tgt = rna_tgt.reshape(-1)

            loss = criterion(logits, rna_tgt)
            total_loss += loss.item()

            valid_tokens = (rna_tgt != pad_token_id).sum().item() # increase by the number of valid (non-padding) tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    model.train()
    return avg_loss, perplexity


def load_config(config_path: str):
    """
    Load YAML configuration file.

    Args:
        config_path (str): Path to the config.yaml file.
    
    Returns:
        dict: Configuration dictionary with all settings.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config
