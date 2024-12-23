import torch
import torch.nn as nn
import fm
from typing import Callable

class AttentionPool(nn.Module):
    """
    Attention pooling, adapted from AttentionPool2d:
    https://benjaminwarner.dev/2022/07/14/tinkering-with-attention-pooling

    Input shape: [B, T, d_model]
      B: batch size
      T: sequence length
      d_model: embedding dimension
    """
    def __init__(self,
        d_model:int,
        bias:bool=True,
        norm:Callable[[int], nn.Module]=nn.LayerNorm
    ):
        super().__init__()

        self.norm = norm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.vk = nn.Linear(d_model, d_model * 2, bias=bias)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, cls_q: torch.Tensor, mask: torch.Tensor = None):
        """
        x: [B, T, d_model]
        cls_q: [1, d_model] (the trainable class query vector)
        mask: [B, T] (boolean mask; True for valid positions, False for padding)
        """
        x = self.norm(x)  # [B, T, d_model]
        B, T, C = x.shape

        q = self.q(cls_q.expand(B, -1, -1))  # [B, 1, d_model]
        k, v = self.vk(x).reshape(B, T, 2, C).permute(2, 0, 1, 3).chunk(2, dim=0)
        k, v = k.squeeze(0), v.squeeze(0)  # [B, T, C], [B, T, C]

        # attn scores
        attn = torch.matmul(q, k.transpose(-2, -1))  # [B, 1, T]

        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1), float("-inf"))

        attn = attn.softmax(dim=-1)
        x = torch.matmul(attn, v).reshape(B, C)  # [B, C]

        return self.proj(x)


class LearnedAggregationSandwich(nn.Module):
    """
    Adaptation of Learned Aggregation, referencing the style:
    https://arxiv.org/abs/2112.13692
    Adapted from:
    https://benjaminwarner.dev/2022/07/14/tinkering-with-attention-pooling
    
    Input shape: [B, T, d_model]
    Output shape: [B, d_model] (one “attention-pooled” representation)
    """
    def __init__(self,
        ni:int,
        attn_bias:bool=True,
        ffn_expand:int|float=3,
        norm:Callable[[int], nn.Module]=nn.LayerNorm,
        act_cls:Callable[[None], nn.Module]=nn.GELU,
    ):
        super().__init__()
        
        # Two learnable scaling parameters
        self.gamma_1 = nn.Parameter(1e-4 * torch.ones(ni))
        self.gamma_2 = nn.Parameter(1e-4 * torch.ones(ni))
        
        self.cls_q = nn.Parameter(torch.zeros([1, ni]))
        
        self.attn = AttentionPool(ni, bias=attn_bias, norm=norm)
        
        self.norm1 = norm(ni)
        self.norm2 = norm(ni)
        
        # Feed-forward “sandwich”
        self.ffn = nn.Sequential(
            nn.Linear(ni, int(ni*ffn_expand)),
            act_cls(),
            norm(int(ni*ffn_expand)),
            nn.Linear(int(ni*ffn_expand), ni)
        )
        
        # Initialize cls_q
        nn.init.trunc_normal_(self.cls_q, std=0.02)
        # Initialize linear layers, etc.
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x: [B, T, d_model]
        Returns: [B, d_model], the “pooled” representation.
        """
        # 1) Attention pooling
        #    a) We do an attention-based pooling with cls_q
        #    b) Then add them together with the residual scaling gamma_1
        attn_out = self.attn(x, self.cls_q, mask)  # [B, d_model]
        x = self.cls_q + self.gamma_1 * self.norm1(attn_out)  # [B, d_model] broadcast-add

        # 2) Feed-forward
        #    (cls_q plus output from feed-forward with residual scaling gamma_2)
        x = x + self.gamma_2 * self.ffn(self.norm2(x))
        return x  # [B, d_model]

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class LengthPredictionHead(nn.Module):
    def __init__(self, d_model: int = 1152, 
                 num_classes: int = 50, 
                 ffn_expand: int|float = 3,
                 dropout: float = 0.0):
        super(LengthPredictionHead, self).__init__()
        self.pooler = LearnedAggregationSandwich(
            ni=d_model, ffn_expand=ffn_expand, norm=nn.LayerNorm
        )

        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: Input tensor of shape [B, T, d_model]
        Returns:
            logits: Output of shape [B, num_classes]
        """
        # Pooling with Learned Aggregation
        x = self.pooler(x, mask)  # -> [B, d_model]
        x = self.dropout(x)
        
        # Classification layer
        logits = self.classifier(x)  # -> [B, num_classes]
        return logits


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
        rna = self.layer_norm(rna) # apply layer norm here because GPT layers don't apply layer norm at the end, but at the begining
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
                 rnafm_checkpoint_path: str = "./fm/weights/rnafm.pt",
                 vocab_size: int = 25):
        """
        d_protein: Dimension of protein embeddings
        d_model: Dimension of model embeddings
        num_heads: Number of attention heads in cross attention blocks
        dropout: Dropout probability
        rnafm_ckpt_path: Path to the RNAFM checkpoint
        vocab_size: Number of tokens in the output vocabulary
        """
        super().__init__()
        self.rnafm, self.alphabet = fm.pretrained.rna_fm_t12()
        self.rnafm_dropout = nn.Dropout(rnafm_dropout)

        for param in self.rnafm.parameters():
            param.requires_grad = True # full finetuning

        num_layers = len(self.rnafm.layers) # number of RNA-FM layers
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(d_protein, d_model, num_heads, rllm_dropout)
            for _ in range(num_layers - 1) # very last layer is in between the final two RNA-FM layers
        ])
        self.layer_norm_blocks = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers) # layer norm after each cross-attention block
        ])


    def forward(self,
                protein_emb: torch.Tensor, 
                rna_emb: torch.Tensor, 
                protein_mask: torch.Tensor = None, 
                rna_mask: torch.Tensor = None) -> torch.Tensor:
        """
        protein_emb: [B, prot_len, d_protein]
        rna_ids: [B, rna_len]
        protein_mask: [B, prot_len]
        rna_mask: [B, rna_len]
        """
        # Generate RNA embeddings from RNA-FM
        for layer_idx, layer in enumerate(self.rnafm.layers):
            res = layer(rna_emb)
            rna_emb = self.rnafm_dropout(res)
            rna_emb = self.cross_attention_blocks[layer_idx](protein_emb, rna_emb, protein_mask, rna_mask)
            rna_emb = self.layer_norm_blocks[layer_idx](rna_emb + res)

        return rna_emb


# class RLLM(nn.Module):
#     def __init__(self, d_protein: int = 1536, 
#                  d_model: int = 768, num_heads: int = 8,
#                  rllm_dropout: float = 0.0,
#                  gpt_dropout: float = 0.0,
#                  gpt_checkpoint_path: str = "./GenerRNA/checkpoint.pt",
#                  vocab_size: int = 1024 + 1
#                 ):
#         """
#         d_protein: Dimension of protein embeddings
#         d_model: Dimension of model embeddings
#         num_heads: Number of attention heads in cross attention blocks
#         dropout: Dropout probability
#         gpt_ckpt_path: Path to the GenerRNA checkpoint
#         vocab_size: Number of tokens in the output vocabulary
#         """
#         super().__init__()
#         self.gpt, self.gpt_args = self.initialize_gpt(gpt_dropout, gpt_checkpoint_path, vocab_size)

#         assert self.gpt_args["n_embd"] == d_model, "Model dimension must match pretrained GPT embedding dimension"
#         assert vocab_size == self.gpt_args["vocab_size"] + 1, "<PAD> token must be added to the vocabulary"

#         self.gpt_args["frozen_layers"] = [i for i in range(self.gpt_args["n_layer"] - (self.gpt_args["n_layer"] // 2))] # indices of layers to freeze
        
#         # freeze first "frozen_layers" layers of GPT
#         for idx, layer in enumerate(self.gpt.transformer.h):
#             if idx in self.gpt_args["frozen_layers"]:
#                 for param in layer.parameters():
#                     param.requires_grad = False

#         num_layers = len(self.gpt.transformer.h) - len(self.gpt_args["frozen_layers"]) # number of layer-norm + cross-attention blocks
#         self.cross_attention_blocks = nn.ModuleList([
#             CrossAttentionBlock(d_protein, d_model, num_heads, rllm_dropout)
#             for _ in range(num_layers)
#         ])

#     def forward(self,
#                 protein_emb: torch.Tensor, 
#                 rna_ids: torch.Tensor, 
#                 protein_mask: torch.Tensor = None, 
#                 rna_mask: torch.Tensor = None) -> torch.Tensor:
#         """
#         protein_emb: [B, prot_len, d_protein]
#         rna_ids: [B, rna_len]
#         protein_mask: [B, prot_len]
#         rna_mask: [B, rna_len]
#         """
#         # Generate RNA embeddings from GPT
#         device = protein_emb.device
#         _, t = rna_ids.size()
#         assert t <= self.gpt_args['block_size'], f"Cannot forward sequence of length {t}, block size is only {self.gpt_args['block_size']}"

#         pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t), t: RNA sequence length

#         tok_emb = self.gpt.transformer.wte(rna_ids) # token embeddings of shape (b, t, n_embd)
#         pos_emb = self.gpt.transformer.wpe(pos) # position embeddings of shape (t, n_embd)

#         x = self.gpt.transformer.drop(tok_emb + pos_emb)

#         for block_idx, block in enumerate(self.gpt.transformer.h):
#             if block_idx in self.gpt_args["frozen_layers"]: 
#                 x = block(x) # frozen layer, keep GPT layer as is
#             else:
#                 # first self-attention, then cross-attention
#                 idx = block_idx - len(self.gpt_args["frozen_layers"]) # cross-attention block index
#                 cross_attn_block = self.cross_attention_blocks[idx]

#                 x = block(x)
#                 x = x + cross_attn_block(protein_emb, x, protein_mask, rna_mask)

#         x = self.gpt.transformer.ln_f(x) # Apply layer normalization to the final output
#         logits = self.gpt.lm_head(x)

#         return logits
    

#     @classmethod
#     def initialize_gpt(cls, 
#                        dropout: float = 0.0, 
#                        gpt_checkpoint_path: str = "./GenerRNA/checkpoint.pt",
#                        vocab_size: int = 1024 + 1): # +1 for <PAD> token
#         """
#         Initialize the GPT decoder with new vocab size
#         """
#         # Initialize pretrained GPT weights and extend with new vocab size (1024 + <PAD>)
#         checkpoint = torch.load(gpt_checkpoint_path)
#         args = checkpoint['model_args']
#         args["dropout"] = dropout

#         conf = GPTConfig(**args)
#         model = GPT(conf)

#         state_dict = checkpoint['model']
#         unwanted_prefix = '_orig_mod.'
#         for k,v in list(state_dict.items()):
#             if k.startswith(unwanted_prefix):
#                 state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
#         model.load_state_dict(state_dict) # load pretrained weights

#         # Extended the vocabulary size => extend the embedding and lm_head layers
#         old_vocab_size, embedding_dim = model.transformer.wte.weight.size()
#         new_wte = nn.Embedding(vocab_size, embedding_dim)
#         new_lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

#         # Copy existing weights
#         new_wte.weight.data[:old_vocab_size] = model.transformer.wte.weight.data
#         new_lm_head.weight.data[:old_vocab_size] = model.lm_head.weight.data # TODO: Is this necessary?

#         # Initialize new weights for the pad token
#         torch.nn.init.normal_(new_wte.weight.data[old_vocab_size:], mean=0.0, std=0.02)
#         torch.nn.init.normal_(new_lm_head.weight.data[old_vocab_size:], mean=0.0, std=0.02) # TODO: Is this necessary?

#         # Replace the old layers
#         model.transformer.wte = new_wte
#         model.lm_head = new_lm_head
#         model.lm_head.weight = model.transformer.wte.weight  # parameters are shared

#         return model, args

#     def get_trainable_parameters(self, 
#                              cross_attn: bool = False, 
#                              layer_norm: bool = False, 
#                              gpt: bool = False):
#         """
#         Returns trainable parameters of the model.

#         Args:
#             cross_attn (bool): Whether to include cross-attention block parameters.
#             layer_norm (bool): Whether to include layer norm block parameters.
#             gpt (bool): Whether to include unfrozen GPT layer parameters.

#         Returns:
#             List of trainable parameters.
#         """
#         params = []

#         if cross_attn:
#             params += list(self.cross_attention_blocks.parameters())

#         if gpt:
#             params += [param for param in self.gpt.parameters() if param.requires_grad]

#         return params