import torch
import os
from torch.utils.data import Dataset
from fm.data import Alphabet, BatchConverter

from constants import RNA_LENGTH_CLUSTERS

class ProteinRNADataset(Dataset):
    def __init__(self, pairs_file: str, 
                 protein_folder: str, 
                 alphabet: Alphabet):
        """
        Args:
            pairs_file: Path to 'train.txt'
            protein_folder: Path to protein embeddings
            rna_folder: Path to RNA embeddings
            tokenizer: Function that tokenizes RNA sequences into indices
        """

        self.protein_folder = protein_folder
        self.alphabet = alphabet

        # assuming pairs file has the following format:
        # >{gene_name}
        # {protein_sequence}${rna_sequence}
        self.pairs = []
        with open(pairs_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) % 2 == 0, "Invalid pairs file"
            for i in range(0, len(lines), 2):  # Every two lines
                gene_name = lines[i].strip()[1:]
                rna_seq = lines[i+1].strip().split('$')[1]
                
                self.pairs.append({
                    "gene_name": gene_name,
                    "rna_seq": rna_seq
                })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):        
        pair = self.pairs[idx]
        
        protein_file = os.path.join(self.protein_folder, f"{pair['gene_name']}.pt")
        protein_emb = torch.load(protein_file, map_location="cpu").squeeze(0)  # Shape: [prot_len, d_protein]

        return {
            "protein": protein_emb, # embedding shape like [prot_len, d_protein]
            "rna": pair["rna_seq"],
        }

    
def collate_fn(batch, tokenizer: BatchConverter):
    protein_lens = [item['protein'].shape[0] for item in batch]
    max_protein_len = max(protein_lens)

    # pad protein embeddings with zeros
    protein_padded = torch.stack([
        torch.cat([item['protein'], torch.zeros(max_protein_len - item['protein'].shape[0], item['protein'].shape[1])])
        for item in batch
    ]) # Shape: [batch_size, max_protein_len, d_protein]
    protein_padding_mask = torch.tensor([[False] * l + [True] * (max_protein_len - l) for l in protein_lens])  # Shape: [batch_size, max_protein_len]

    rna_sequences = [item['rna'] for item in batch]

    rna_sequences = [(str(i), rna) for i, rna in enumerate(rna_sequences)] # format that BatchConverter expects
    
    _, _, tokens = tokenizer(rna_sequences)

    rna_padding_mask = tokens == tokenizer.alphabet.padding_idx

    return {
        "protein": protein_padded,          # [batch_size, max_protein_len, d_protein]
        "protein_mask": protein_padding_mask,  # [batch_size, max_protein_len]
        "rna": tokens,                  # [batch_size, max_rna_len]
        "rna_mask": rna_padding_mask,          # [batch_size, max_rna_len]
    }


class ProteinRNALengthDataset(Dataset):
    def __init__(self, pairs_file: str, 
                 protein_folder: str):
        """
        Args:
            pairs_file: Path to 'train.txt'
            protein_folder: Path to protein embeddings
        """

        self.protein_folder = protein_folder

        # assuming pairs file has the following format:
        # >{gene_name}
        # {protein_sequence}${rna_sequence}
        self.pairs = []
        with open(pairs_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) % 2 == 0, "Invalid pairs file"
            for i in range(0, len(lines), 2):  # Every two lines
                gene_name = lines[i].strip()[1:]
                rna_seq = lines[i+1].strip().split('$')[1]

                # assign RNA length to a cluster
                length = len(rna_seq)
                label = None
                for cluster_id, (min_length, max_length), _ in RNA_LENGTH_CLUSTERS:
                    if min_length <= length <= max_length:
                        label = cluster_id
                        break
                assert label is not None, f"RNA length {length} not in any cluster"
                
                self.pairs.append({
                    "gene_name": gene_name,
                    "label": label
                })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):        
        pair = self.pairs[idx]
        
        protein_file = os.path.join(self.protein_folder, f"{pair['gene_name']}.pt")
        protein_emb = torch.load(protein_file, map_location="cpu").squeeze(0)  # Shape: [prot_len, d_protein]

        return {
            "protein": protein_emb, # embedding shape like [prot_len, d_protein]
            "label": pair["label"],
        }
    
def collate_fn_length(batch):
    protein_lens = [item['protein'].shape[0] for item in batch]
    max_protein_len = max(protein_lens)

    # pad protein embeddings with zeros
    protein_padded = torch.stack([
        torch.cat([item['protein'], torch.zeros(max_protein_len - item['protein'].shape[0], item['protein'].shape[1])])
        for item in batch
    ]) # Shape: [batch_size, max_protein_len, d_protein]
    protein_padding_mask = torch.tensor([[False] * l + [True] * (max_protein_len - l) for l in protein_lens])  # Shape: [batch_size, max_protein_len]

    labels = [item['label'] for item in batch]

    return {
        "protein": protein_padded,          # [batch_size, max_protein_len, d_protein]
        "protein_mask": protein_padding_mask,  # [batch_size, max_protein_len]
        "labels": torch.tensor(labels),                  # [batch_size]
    }