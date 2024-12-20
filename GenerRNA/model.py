import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

        
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, strategy='sampling', beam_size=3, eos_token_id=0, repetition_penalty=1.0):
        assert strategy in ['greedy_search', 'sampling', 'top_k', 'beam_search']

        batch_size = idx.size(0)
        if strategy == 'beam_search':
            # Initialize beams
            beam_seqs = [idx.clone() for _ in range(beam_size)]
            beam_scores = torch.zeros((batch_size, beam_size), device=idx.device)
            completed_seqs = []

            for _ in range(max_new_tokens):
                all_candidates = []
                for i in range(beam_size):
                    idx_cond = beam_seqs[i] if beam_seqs[i].size(1) <= self.config.block_size else beam_seqs[i][:, -self.config.block_size:]
                    logits, _ = self(idx_cond)
                    logits = logits[:, -1, :] / temperature
                    if repetition_penalty != 1.0:
                        for j in range(idx_cond.size(1)):
                            logits[:, idx_cond[:, j]] /= repetition_penalty
                    probs = F.log_softmax(logits, dim=-1)
                    scores, indices = torch.topk(probs, beam_size, dim=-1)

                    for j in range(beam_size):
                        candidate_seq = torch.cat([beam_seqs[i], indices[:, j:j+1]], dim=1)
                        candidate_score = beam_scores[:, i] + scores[:, j]
                        if indices[0, j] == eos_token_id:
                            completed_seqs.append((candidate_score, candidate_seq))
                        else:
                            all_candidates.append((candidate_score, candidate_seq))

                # add random noise when sorting beacause that generated sequences of beam_search remain unchanged if they have the same prefix
                all_candidates.sort(key=lambda x: x[0].mean().item() + torch.rand(1).item() * 5e-1, reverse=True)

                beam_seqs = [all_candidates[i][1] for i in range(min(beam_size, len(all_candidates)))]
                beam_scores = torch.stack([all_candidates[i][0] for i in range(min(beam_size, len(all_candidates)))], dim=1)
                if len(completed_seqs) >= beam_size:
                    break

            if not completed_seqs:
                completed_seqs = [(beam_scores[:, i], beam_seqs[i]) for i in range(beam_size)]

            completed_seqs.sort(key=lambda x: x[0].mean().item(), reverse=True)
            return completed_seqs[0][1]


        else:
            for _ in range(max_new_tokens):
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature

                if repetition_penalty != 1.0:
                    for i in range(idx.size(0)):
                        for j in range(idx.size(1)):
                            logits[i, idx[i, j]] /= repetition_penalty

                if strategy == 'greedy_search':
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                
                elif strategy == 'sampling':
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                
                elif strategy == 'top_k':
                    if top_k is not None:
                        logits, indices = torch.topk(logits, min(top_k, logits.size(-1)))
                        probs = F.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        idx_next = torch.gather(indices, dim=-1, index=idx_next)

                if idx_next == eos_token_id:
                    break
                idx = torch.cat((idx, idx_next), dim=1)

        return idx if idx[0][0] != eos_token_id else idx[:, 1:]
