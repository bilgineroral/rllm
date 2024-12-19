import torch
import torch.nn as nn
from GenerRNA.model import GPT, GPTConfig
import einops

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
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                protein_emb: torch.Tensor, 
                rna_emb: torch.Tensor, 
                protein_mask: torch.Tensor = None, 
                rna_mask: torch.Tensor = None) -> torch.Tensor:
        """
        protein_emb: [B, prot_len, d_protein]
        rna_emb: [B, rna_len, d_model]
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
        # TODO: This might be unnecessary, but doesn't harm either
        if rna_mask is not None:
            rna_mask = rna_mask.unsqueeze(-1).expand(attn_output.shape)  # [B, rna_len, d_model]
            attn_output = attn_output.masked_fill(rna_mask, 0.0)

        return attn_output


class RLLM(nn.Module):
    def __init__(self, d_protein: int = 1536, 
                 d_model: int = 768, num_heads: int = 8,
                 rllm_dropout: float = 0.0,
                 gpt_dropout: float = 0.0,
                 gpt_checkpoint_path: str = "./GenerRNA/checkpoint.pt",
                 vocab_size: int = 1024 + 1
                ):
        """
        d_protein: Dimension of protein embeddings
        d_model: Dimension of model embeddings
        num_heads: Number of attention heads in cross attention blocks
        dropout: Dropout probability
        gpt_ckpt_path: Path to the GenerRNA checkpoint
        vocab_size: Number of tokens in the output vocabulary
        """
        super().__init__()
        self.gpt, self.gpt_args = self.initialize_gpt(gpt_dropout, gpt_checkpoint_path, vocab_size)

        assert self.gpt_args["n_embd"] == d_model, "Model dimension must match pretrained GPT embedding dimension"
        assert vocab_size == self.gpt_args["vocab_size"] + 1, "<PAD> token must be added to the vocabulary"

        self.gpt_args["frozen_layers"] = [i for i in range(self.gpt_args["n_layer"] // 2)] # indices of layers to freeze
        
        # freeze first 12 layers of GPT
        for idx, layer in enumerate(self.gpt.transformer.h):
            if idx not in self.gpt_args["frozen_layers"]:
                for param in layer.parameters():
                    param.requires_grad = False

        num_layers = len(self.gpt.transformer.h) - len(self.gpt_args["frozen_layers"]) # number of cross-attention / layer-norm blocks
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(d_protein, d_model, num_heads, rllm_dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm_blocks = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])


    def forward(self,
                protein_emb: torch.Tensor, 
                rna_ids: torch.Tensor, 
                protein_mask: torch.Tensor = None, 
                rna_mask: torch.Tensor = None) -> torch.Tensor:
        """
        protein_emb: [B, prot_len, d_protein]
        rna_ids: [B, rna_len]
        protein_mask: [B, prot_len]
        rna_mask: [B, rna_len]
        """
        # Generate RNA embeddings from GPT
        device = protein_emb.device
        _, t = rna_ids.size()
        assert t <= self.gpt_args['block_size'], f"Cannot forward sequence of length {t}, block size is only {self.gpt_args['block_size']}"

        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t), t: RNA sequence length

        tok_emb = self.gpt.transformer.wte(rna_ids) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.gpt.transformer.wpe(pos) # position embeddings of shape (t, n_embd)

        x = self.gpt.transformer.drop(tok_emb + pos_emb)

        for block_idx, block in enumerate(self.gpt.transformer.h):
            if block_idx in self.gpt_args["frozen_layers"]: 
                x = block(x, mlp=False)
                x = block(x, mlp=True)
            else:
                # first self-attention, then cross-attention
                idx = block_idx - len(self.gpt_args["frozen_layers"]) # cross-attention block index
                cross_attn_block = self.cross_attention_blocks[idx]
                layer_norm_block = self.layer_norm_blocks[idx]

                x = block(x, mlp=False)
                x = x + cross_attn_block(protein_emb, x, protein_mask, rna_mask)
                x = layer_norm_block(x)
                x = block(x, mlp=True)

        x = self.gpt.transformer.ln_f(x) # Apply layer normalization to the final output
        logits = self.gpt.lm_head(x)

        return logits
    

    @classmethod
    def initialize_gpt(cls, 
                       dropout: float = 0.0, 
                       gpt_checkpoint_path: str = "./GenerRNA/checkpoint.pt",
                       vocab_size: int = 1025):
        """
        Initialize the GPT decoder with new vocab size
        """
        # Initialize pretrained GPT weights and extend with new vocab size (1024 + <PAD>)
        checkpoint = torch.load(gpt_checkpoint_path)
        args = checkpoint['model_args']
        args["dropout"] = dropout

        conf = GPTConfig(**args)
        model = GPT(conf)

        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict) # load pretrained weights

        # Extended the vocabulary size => extend the embedding and lm_head layers
        old_vocab_size, embedding_dim = model.transformer.wte.weight.size()
        new_wte = nn.Embedding(vocab_size, embedding_dim)
        new_lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

        # Copy existing weights
        new_wte.weight.data[:old_vocab_size] = model.transformer.wte.weight.data
        new_lm_head.weight.data[:old_vocab_size] = model.lm_head.weight.data # TODO: Is this necessary?

        # Initialize new weights for the pad token
        torch.nn.init.normal_(new_wte.weight.data[old_vocab_size:], mean=0.0, std=0.02)
        torch.nn.init.normal_(new_lm_head.weight.data[old_vocab_size:], mean=0.0, std=0.02) # TODO: Is this necessary?

        # Replace the old layers
        model.transformer.wte = new_wte
        model.lm_head = new_lm_head
        model.lm_head.weight = model.transformer.wte.weight  # parameters are shared

        return model, args

    def get_trainable_parameters(self, 
                             cross_attn: bool = False, 
                             layer_norm: bool = False, 
                             gpt: bool = False):
        """
        Returns trainable parameters of the model.

        Args:
            cross_attn (bool): Whether to include cross-attention block parameters.
            layer_norm (bool): Whether to include layer norm block parameters.
            gpt (bool): Whether to include unfrozen GPT layer parameters.

        Returns:
            List of trainable parameters.
        """
        params = []

        if cross_attn:
            params += list(self.cross_attention_blocks.parameters())

        if layer_norm:
            params += list(self.layer_norm_blocks.parameters())

        if gpt:
            params += [param for param in self.gpt.parameters() if param.requires_grad]

        return params