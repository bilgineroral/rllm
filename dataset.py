import torch
import os
from torch.utils.data import Dataset

from transformers import AutoTokenizer

class ProteinRNADataset(Dataset):
    def __init__(self, pairs_file: str, 
                 protein_folder: str, 
                 tokenizer: AutoTokenizer = None,
                 offset: int = 0):
        """
        Args:
            pairs_file: Path to 'train.txt'
            protein_folder: Path to protein embeddings
            rna_folder: Path to RNA embeddings
            tokenizer: Function that tokenizes RNA sequences into indices
            offset: Number of pairs to skip, useful for resuming sampling from the dataloader
        """
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

        self.protein_folder = protein_folder
        self.tokenizer = tokenizer
        self.offset = offset

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
        
        # Skip pairs based on given offset
        self.pairs = self.pairs[self.offset:] # https://stackoverflow.com/a/67073875

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

    
def collate_fn(batch, tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

    protein_lens = [item['protein'].shape[0] for item in batch]
    max_protein_len = max(protein_lens)

    # pad protein embeddings with zeros
    protein_padded = torch.stack([
        torch.cat([item['protein'], torch.zeros(max_protein_len - item['protein'].shape[0], item['protein'].shape[1])])
        for item in batch
    ]) # Shape: [batch_size, max_protein_len, d_protein]
    protein_padding_mask = torch.tensor([[False] * l + [True] * (max_protein_len - l) for l in protein_lens])  # Shape: [batch_size, max_protein_len]

    rna_sequences = [item['rna'] for item in batch]
    tokens = tokenizer(rna_sequences, padding="longest", return_tensors="pt")
    rna_ids = tokens["input_ids"]
    rna_mask = tokens["attention_mask"]
    rna_mask = rna_mask == 0  # Convert to causal mask

    return {
        "protein": protein_padded,          # [batch_size, max_protein_len, d_protein]
        "protein_mask": protein_padding_mask,  # [batch_size, max_protein_len]
        "rna": rna_ids,                  # [batch_size, max_rna_len, d_rna]
        "rna_mask": rna_mask,          # [batch_size, max_rna_len]
    }