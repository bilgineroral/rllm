import torch
import torch.nn as nn
import fm

class AttentionPooling(nn.Module):
    def __init__(self, d_model: int = 1152):
        super().__init__()
        self.attention_weights = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x: [B, seq_len, d_model] - Input sequence embeddings
        mask: [B, seq_len] - Boolean mask (True for padding tokens, False otherwise)
        """
        # Compute attention scores: [B, seq_len, 1]
        scores = self.attention_weights(x)  # Raw scores

        # Apply mask: Assign large negative values to padding positions
        if mask is not None:
            mask = mask.unsqueeze(-1)  # [B, seq_len, 1]
            scores = scores.masked_fill(mask, float("-inf"))  # Set scores of padding tokens to -inf

        # Compute normalized weights: [B, seq_len, 1]
        weights = torch.softmax(scores, dim=1)

        # Compute weighted sum of inputs: [B, d_model]
        output = (x * weights).sum(dim=1)
        return output
    

class LengthPredictionHead(nn.Module):
    def __init__(self, d_model: int = 1152, 
                 num_classes: int = 50, 
                 dropout: float = 0.0):
        super(LengthPredictionHead, self).__init__()
        self.attn_pool = AttentionPooling(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(4*d_model),
            nn.Linear(4*d_model, 2*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(2*d_model),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4*d_model, d_model),  # 4*d_model because of concatenation
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: Input tensor of shape [B, T, d_model]
        Returns:
            logits: Output of shape [B, num_classes]
        """
        # Pooling with Learned Aggregation
        avg_pool = x.mean(dim=1)  # [B, d_model]
        max_pool, _ = x.max(dim=1) # [B, d_model]
        attn_pool = self.attn_pool(x, mask) # [B, d_model]
        
        x = self.ffn(attn_pool) # [B, 2*d_model]
        x = torch.cat([x, avg_pool, max_pool], dim=-1) # [B, 4*d_model]
        return self.classifier(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_protein: int = 1536, 
                 d_model: int = 768, num_heads: int = 8,
                 dropout: float = 0.0
                ):
        """
        d_protein: Dimension of protein embeddings
        d_model: Dimension of model embeddings
        num_heads: Number of attention heads
        vocab_size: Number of tokens in the output vocabulary
        """
        super().__init__()
        self.protein_proj = nn.Linear(d_protein, d_model)
        self.rna_proj = nn.Linear(d_model, d_model)
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                protein: torch.Tensor, 
                rna: torch.Tensor, 
                protein_mask: torch.Tensor = None, 
                rna_mask: torch.Tensor = None) -> torch.Tensor:
        """
        protein: [B, prot_len, d_protein]
        rna: [B, rna_len, d_model]
        protein_mask: [B, prot_len]
        rna_mask: [B, rna_len]
        """
        # Project protein and RNA embeddings to a common dimension        
        rna = self.layer_norm(rna) # Layer norm before projection
        rna_proj = self.dropout(self.rna_proj(rna))              # [B, rna_len, d_model]
        protein_proj = self.dropout(self.protein_proj(protein))  # [B, prot_len, d_model]

        # Cross Attention: RNA queries protein
        attn_output, _ = self.cross_attention(
            query=rna_proj, key=protein_proj, value=protein_proj,
            key_padding_mask=protein_mask
        )

        return attn_output
    

class RLLM(nn.Module):
    def __init__(self, d_protein: int = 1152,
                 d_model: int = 640, num_heads: int = 8,
                 rllm_dropout: float = 0.0,
                 rnafm_dropout: float = 0.0,
                 rnafm_checkpoint_path: str = "./fm/weights/rnafm.pt"):
        """
        d_protein: Dimension of protein embeddings
        d_model: Dimension of model embeddings
        num_heads: Number of attention heads in cross attention blocks
        dropout: Dropout probability
        rnafm_ckpt_path: Path to the RNAFM checkpoint
        vocab_size: Number of tokens in the output vocabulary
        """
        super().__init__()
        self.rnafm, self.alphabet = fm.pretrained.rna_fm_t12(rnafm_checkpoint_path)
        self.rnafm_dropout = nn.Dropout(rnafm_dropout)

        for param in self.rnafm.parameters():
            param.requires_grad = True # full finetuning

        num_layers = len(self.rnafm.layers) # number of RNA-FM layers
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(d_protein, d_model, num_heads, rllm_dropout)
            for _ in range(num_layers) # cross-attention block after each RNA-FM layer
        ])
        self.layer_norm_blocks = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers) # layer norm after each cross-attention block
        ])


    def forward(self,
                protein_emb: torch.Tensor, 
                rna_ids: torch.Tensor, 
                protein_mask: torch.Tensor = None, # protein padding mask 
                rna_mask: torch.Tensor = None, # RNA padding mask
                masked_tokens: torch.Tensor = None # masked tokens for language modeling
            ) -> torch.Tensor: 
        """
        protein_emb: [B, prot_len, d_protein]
        rna_ids: [B, rna_len] - RNA tokens
        protein_mask: [B, prot_len] - protein padding mask
        rna_mask: [B, rna_len] - RNA padding mask
        masked_tokens: [B, rna_len] - masked tokens for RNA language modeling
        """

        x = self.rnafm.embed_scale * self.rnafm.embed_tokens(rna_ids) # [B, rna_len, d_model]
        x = x + self.rnafm.embed_positions(rna_ids) # [B, rna_len, d_model]

        # zero out padding tokens, following the RNA-FM implementation
        if rna_mask is not None:
            x = x.masked_fill((rna_ids == self.rnafm.padding_idx).unsqueeze(-1), 0.0)

            if not rna_mask.any():
                rna_mask = None

        # Generate RNA embeddings from RNA-FM
        for layer_idx, layer in enumerate(self.rnafm.layers):
            x = x.transpose(0, 1) # (B, T, E) => (T, B, E) conversion due to RNA-FM implementation
            x, _ = layer(
                x, self_attn_padding_mask=rna_mask, need_head_weights=False
            )
            x = self.rnafm_dropout(x)
            x = x.transpose(0, 1) # (T, B, E) => (B, T, E) convert back
            residual = x
            x = self.cross_attention_blocks[layer_idx](protein_emb, x, protein_mask, rna_mask)
            x = x + residual
            x = self.layer_norm_blocks[layer_idx](x)
        
        logits = self.rnafm.lm_head(x, masked_tokens)
        return logits # [B, rna_len, vocab_size]


    def get_trainable_parameters(self, 
                             cross_attn: bool = False, 
                             layer_norm: bool = False, 
                             rnafm: bool = False):
        """
        Returns trainable parameters of the model.

        Args:
            cross_attn (bool): Whether to include cross-attention block parameters.
            layer_norm (bool): Whether to include layer norm block parameters.
            rnafm (bool): Whether to include RNA-FM layer parameters.

        Returns:
            List of trainable parameters.
        """
        params = []

        if cross_attn:
            params += list(self.cross_attention_blocks.parameters())

        if rnafm:
            params += list(self.rnafm.parameters())

        if layer_norm:
            params += list(self.layer_norm_blocks.parameters())

        return params