import torch
import torch.nn as nn
import einops

class CrossAttentionBlock(nn.Module):
    def __init__(self, d_protein: int = 1536, 
                 d_rna: int = 768, 
                 d_model: int = 768, num_heads: int = 8,
                 dropout: float = 0.1
                ):
        """
        d_protein: Dimension of protein embeddings
        d_rna: Dimension of RNA embeddings
        d_model: Dimension of model embeddings
        num_heads: Number of attention heads
        vocab_size: Number of tokens in the output vocabulary
        """
        super().__init__()
        self.protein_proj = nn.Linear(d_protein, d_model)
        self.rna_proj = nn.Linear(d_rna, d_model)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, protein_emb: torch.Tensor, 
                rna_emb: torch.Tensor, 
                protein_mask: torch.Tensor = None, 
                rna_mask: torch.Tensor = None) -> torch.Tensor:
        """
        protein_emb: [B, prot_len, d_protein]
        rna_emb: [B, rna_len, d_rna]
        protein_mask: [B, prot_len]
        rna_mask: [B, rna_len]
        """
        # Project protein and RNA embeddings to a common dimension
        protein_proj = self.dropout(self.protein_proj(protein_emb))  # [B, prot_len, d_model]
        rna_proj = self.dropout(self.rna_proj(rna_emb))              # [B, rna_len, d_model]

        # Cross Attention: RNA queries protein
        attn_output, _ = self.cross_attention(
            query=rna_proj, key=protein_proj, value=protein_proj,
            key_padding_mask=protein_mask
        )

        # Mask out padded RNA positions
        if rna_mask is not None:
            rna_mask = rna_mask.unsqueeze(-1)  # [B, rna_len, 1]
            attn_output = attn_output.masked_fill(rna_mask, 0.0)

        # Add & Norm
        output = self.layer_norm(rna_proj + attn_output)
        return output


class ParallelizedCrossAttentionModel(nn.Module):
    def __init__(self, d_rna: int = 768,
                 d_protein: int = 1536,
                 d_model: int = 768,
                 num_heads: int = 8,
                 num_layers: int = 12,
                 dropout: float = 0.1,
                 vocab_size: int = 8):
        super().__init__()
        self.num_layers = num_layers
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(d_protein, d_rna, d_model, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.output_head = MLPOutputHead(d_model=d_model, num_layers=num_layers, vocab_size=vocab_size)
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, rna_emb: torch.Tensor, 
                protein_emb: torch.Tensor, 
                rna_mask: torch.Tensor = None,
                protein_mask: torch.Tensor = None) -> torch.Tensor:
        """
        rna_emb: [batch_size, num_layers, rna_seq_len, d_rna]
        protein_emb: [batch_size, protein_seq_len, d_protein]
        protein_mask: [batch_size, protein_seq_len]
        rna_mask: [batch_size, rna_seq_len]
        """
        num_layers = rna_emb.shape[1]
        assert num_layers == self.num_layers, "Mismatch in number of layers"

        # Fork each cross-attention block
        futures = []
        for i in range(self.num_layers):
            rna_layer_emb = rna_emb[:, i, :, :]  # Shape: [batch_size, rna_seq_len, d_rna]
            futures.append(torch.jit.fork(
                self.cross_attention_blocks[i],
                protein_emb, rna_layer_emb, protein_mask, rna_mask
            ))

        # Gather results
        outputs = [torch.jit.wait(fut) for fut in futures]

        # Stack outputs along the layer dimension
        concatenated_output = torch.stack(outputs, dim=1)  # Shape: [batch_size, num_layers, rna_seq_len, d_model]
        concatenated_output = einops.rearrange(concatenated_output, 'b l r d -> b r (l d)')
        concatenated_output = self.dropout(concatenated_output) 

        logits = self.output_head(concatenated_output)  # Shape: [B, rna_seq_len, vocab_size]
        return logits
    

class MLPOutputHead(nn.Module):
    def __init__(self, d_model, num_layers, vocab_size, dropout=0.1):
        super(MLPOutputHead, self).__init__()
        
        input_dim = d_model * num_layers 
        hidden_dim1 = input_dim // 2
        hidden_dim2 = input_dim // 4 
        self.output_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1), 
            nn.LayerNorm(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(p=dropout), 

            nn.Linear(hidden_dim1, hidden_dim2), 
            nn.LayerNorm(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(hidden_dim2, d_model), 
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(d_model, vocab_size) 
        )
        
    def forward(self, x):
        # x should have shape [batch_size, seq_len, d_model * num_layers]
        return self.output_head(x)