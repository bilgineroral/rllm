import os
import yaml
import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch

from model import RLLM, LengthPredictionHead

def plot(data: list, save_path: str, step: int = 1) -> None:
    """
    Plots loss/perplexity values against iterations and saves the plot to a specified path.
    
    Args:
        data (list of tuples): A list where each tuple contains (value, iteration).
        save_path (str): The file path to save the generated plot.
        step (int): The step size for plotting data points.
    """
    # Unpack loss values and iterations
    loss_values, iterations = zip(*data)
    
    # Reduce the number of points to plot
    reduced_iterations = iterations[::step]
    reduced_loss_values = loss_values[::step]
    
    # Plot the raw loss values
    plt.figure(figsize=(10, 6))
    plt.plot(reduced_iterations, reduced_loss_values, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Loss Value')
    plt.title('Loss vs Iterations (Raw Data)')
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
             device: torch.device
            ) -> float:
    """
    Validates the model on the validation set or a subset of it.
    
    Args:
        model (torch.nn.Module): The model to validate.
        dataloader (torch.utils.data.DataLoader): The validation DataLoader.
        criterion (torch.nn.Module): Loss function (e.g., CrossEntropyLoss).

    Returns:
        float: Validation loss.
        float: Perplexity score.
    """
    assert type(criterion) == torch.nn.CrossEntropyLoss, "Only CrossEntropyLoss is supported for now."
    assert criterion.reduction == "sum", "Reduction should be 'sum' for validation."

    model.eval()
    vocab_size = len(model.alphabet.all_toks)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            protein, protein_padding_mask, masked_rna, mask_indices, gt_tokens, rna_padding_mask = (
                batch["protein"].to(device), # embedding shape like [B, prot_len, d_protein]
                batch["protein_padding_mask"].to(device), # mask shape like [B, prot_len]
                batch["masked_rna"].to(device), # tokenized RNA sequence shape like [B, rna_len]
                batch["mask_indices"].to(device), # mask shape like [B, rna_len]
                batch["ground_truth_tokens"], # ground truth tokens -- list (size: B) of lists
                batch["rna_padding_mask"].to(device) # mask shape like [B, rna_len]
            )

            tokens_mask = torch.zeros(
                masked_rna.size(0), masked_rna.size(1), 
                device=masked_rna.device, dtype=torch.bool
            ) # shape like [B, rna_len]
            tokens_mask.scatter_(1, mask_indices, True) # shape like [B, rna_len], True at masked indices

            logits = model(protein, masked_rna, protein_padding_mask, rna_padding_mask, masked_tokens=tokens_mask)
            logits = logits.reshape(-1, vocab_size)
            gt_tokens = [model.alphabet.get_idx(t) for tokens in gt_tokens for t in tokens]
            gt_tokens = torch.tensor(gt_tokens, device=device, dtype=torch.long)

            loss = criterion(logits, gt_tokens)

            total_loss += loss.item()
            total_tokens += gt_tokens.size(0)

    avg_loss = total_loss / total_tokens

    model.train()
    return avg_loss, math.exp(avg_loss)


def validate_ln(model: LengthPredictionHead, 
             dataloader: torch.utils.data.DataLoader, 
             criterion: torch.nn.Module,
             device: torch.device
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
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            protein, protein_mask, labels = (
                batch["protein"].to(device), # embedding shape like [B, prot_len, d_protein]
                batch["protein_mask"].to(device), # mask shape like [B, prot_len]
                batch["labels"].to(device), # shape like [B]
            )

            logits = model(protein, protein_mask) # shape like [B, num_classes]
            loss = criterion(logits, labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    model.train()
    return avg_loss


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

def parse_data_file(path: str):
    """
    Parse the data file (losses or perplexities) and return a list of tuples.

    Args:
        path (str): Path to the data file.

    Returns:
        list: A list of tuples where each tuple contains (value, iteration).
    """

    data = []
    if os.path.exists(path):    
        with open(path, "r") as file:
            for line in file:
                value, iteration = eval(line)
                data.append((float(value), int(iteration)))

    return data


def mask_tokens(sequence: str, 
                vocab: list = None
                ) -> tuple[str, np.ndarray, list]:
    """
    Apply the 80/10/10 rule to mask tokens in a sequence.
    
    Returns:
        tuple: (masked_sequence, mask_indices, ground_truth_tokens)
            - masked_sequence: string with tokens masked
            - mask_indices: numpy array of masked positions
            - ground_truth_tokens: list of original tokens at masked positions
    """
    if vocab is None:
        vocab = [
            'A', 'C', 'G', 'U', 'R', 'Y', 'K', 
            'M', 'S', 'W', 'B', 'D', 'H', 'V', 'N'
        ]
        
    seq_length = len(sequence)
    num_to_mask = random.randint(1, seq_length)
    mask_indices = np.random.choice(seq_length, num_to_mask, replace=False)
    mask_indices = np.sort(mask_indices)
    
    # Track ground truth tokens
    ground_truth_tokens = [sequence[idx] for idx in mask_indices]
    
    masked_sequence = list(sequence)
    for idx in mask_indices:
        prob = random.random()
        if prob < 0.8:  # 80%: Replace with [MASK]
            masked_sequence[idx] = "-"
        elif prob < 0.9:  # 10%: Replace with random token
            masked_sequence[idx] = random.choice(vocab)
        # else: 10% unchanged

    return "".join(masked_sequence), mask_indices, ground_truth_tokens

