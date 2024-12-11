import torch
import os
from torch.utils.data import Dataset

from tokenizer import RNATokenizer

class ProteinRNADataset(Dataset):
    def __init__(self, pairs_file: str, 
                 protein_folder: str, 
                 rna_folder: str, 
                 tokenizer: RNATokenizer = None):
        """
        Args:
            pairs_file: Path to 'representative_pairs.txt'
            protein_folder: Path to protein embeddings
            rna_folder: Path to RNA embeddings
            tokenizer: Function that tokenizes RNA sequences into indices
        """
        self.protein_folder = protein_folder
        self.rna_folder = rna_folder
        if tokenizer is None:
            tokenizer = RNATokenizer()
        self.tokenizer = tokenizer

        # Parse the pairs file
        self.pairs = []
        with open(pairs_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) % 2 == 0, "Invalid pairs file"
            for i in range(0, len(lines), 2):  # Every two lines
                identifier = lines[i].strip()[1:]
                rna_seq = lines[i+1].strip().split('$')[1]
                pdb_id, prot_chain, rna_chain = identifier.split('_')

                self.pairs.append({
                    "pdb_id": pdb_id,
                    "prot_chain": prot_chain,
                    "rna_chain": rna_chain, 
                    "rna_seq": rna_seq
                })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Retrieve file paths
        pair = self.pairs[idx]
        protein_file = os.path.join(self.protein_folder, f"{pair['pdb_id']}_{pair['prot_chain']}.pt")
        rna_file = os.path.join(self.rna_folder, f"{pair['pdb_id']}_{pair['rna_chain']}.pt")

        # Load embeddings lazily
        protein_emb = torch.load(protein_file).squeeze(0)  # Shape: [prot_len + 2, 1536]
        rna_emb = torch.load(rna_file)
        rna_emb = torch.tensor(rna_emb, dtype=torch.float32)  # Shape: [rna_num_layers (12), rna_len + 2, 768]
        
        # Remove the last token (EOS) from RNA embeddings
        rna_emb = rna_emb[:, :-1, :]  # Shape: [12, seq_len + 1, d_rna]

        # Tokenize RNA sequence and create target
        rna_tokens = self.tokenizer(pair["rna_seq"])  # List of token indices with <SOS> and <EOS>
        rna_input = torch.tensor(rna_tokens[:-1], dtype=torch.long)  # Input
        rna_target = torch.tensor(rna_tokens[1:], dtype=torch.long) # Target (shifted)

        return {
            "protein": protein_emb,
            "rna": rna_emb, # unused when training RNA model
            "rna_input": rna_input, # unused when using pre-generated embeddings
            "rna_target": rna_target
        }
    

def collate_fn(batch, tokenizer=None, device: torch.device = torch.device('cuda')):
    protein_lens = [item['protein'].shape[0] for item in batch]
    max_protein_len = max(protein_lens)

    # pad protein embeddings with zeros
    protein_padded = torch.stack([
        torch.cat([item['protein'].to(device), torch.zeros(max_protein_len - item['protein'].shape[0], item['protein'].shape[1]).to(device)])
        for item in batch
    ]) # Shape: [batch_size, max_protein_len, d_protein]

    rna_lens = [item['rna'].shape[1] for item in batch]
    max_rna_len = max(rna_lens)

    # pad rna embeddings with zeros
    rna_padded = torch.stack([
        torch.cat([item['rna'].to(device), torch.zeros(item['rna'].shape[0], max_rna_len - item['rna'].shape[1], item['rna'].shape[2]).to(device)], dim=1)
        for item in batch
    ])  # Shape: [batch_size, rna_num_layers (12), max_rna_len, d_rna]

    # Target padding
    if tokenizer is None:
        tokenizer = RNATokenizer()
    rna_targets = [item['rna_target'] for item in batch]
    rna_targets_padded, _ = tokenizer.pad_sequences(rna_targets)

    # Create masks
    protein_padding_mask = torch.tensor([[False] * l + [True] * (max_protein_len - l) for l in protein_lens]).to(device)
    rna_padding_mask = torch.tensor([[False] * l + [True] * (max_rna_len - l) for l in rna_lens]).to(device)

    return {
        "protein": protein_padded,          # [batch_size, max_protein_len, d_protein]
        "rna": rna_padded,                  # [batch_size, max_rna_len, d_rna]
        "protein_mask": protein_padding_mask,  # [batch_size, max_protein_len]
        "rna_mask": rna_padding_mask,          # [batch_size, max_rna_len]
        "rna_targets": rna_targets_padded   # [batch_size, max_rna_len]
    }
