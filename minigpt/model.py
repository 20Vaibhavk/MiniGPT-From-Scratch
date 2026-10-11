"""
MiniGPT: a decoder-only transformer, the same family as GPT-2, GPT-3 and ChatGPT.

    token ids
      -> token embedding + position embedding
      -> N x [ LayerNorm -> Multi-Head Attention -> +residual
               LayerNorm -> Feed-Forward          -> +residual ]
      -> final LayerNorm
      -> Linear "language-model head"  -> logits over the vocabulary
"""
import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .config import GPTConfig
from .layers import (GELU, Dropout, Embedding, LayerNorm, Linear,
                     SinusoidalPositionalEncoding, cross_entropy, softmax)


class FeedForward(nn.Module):
    """Position-wise MLP: expand 4x, non-linearity, project back.

    Attention lets tokens *talk to each other*; the feed-forward layer lets
    every token *think on its own* about what it just heard.
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.fc = Linear(n_embd, 4 * n_embd)
        self.act = GELU()
        self.proj = Linear(4 * n_embd, n_embd)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(self.act(self.fc(x))))


class TransformerBlock(nn.Module):
    """One transformer decoder block (pre-norm, as in GPT-2)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.n_embd)
        self.attn = MultiHeadAttention(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout)
        self.ln2 = LayerNorm(cfg.n_embd)
        self.ffwd = FeedForward(cfg.n_embd, cfg.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # communicate
        x = x + self.ffwd(self.ln2(x))   # compute
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = Embedding(cfg.vocab_size, cfg.n_embd)
        if cfg.pos_encoding == "sinusoidal":
            self.position_embedding = SinusoidalPositionalEncoding(cfg.block_size, cfg.n_embd)
        else:
            self.position_embedding = Embedding(cfg.block_size, cfg.n_embd)
        self.dropout = Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.n_embd)
        self.lm_head = Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # GPT-2 trick: shrink the init of layers that write into the residual
        # stream so the signal does not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / (2 * cfg.n_layer) ** 0.5)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        positions = torch.arange(T, device=idx.device)

        x = self.token_embedding(idx) + self.position_embedding(positions)   # (B, T, C)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                            # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def next_token_probs(self, idx, temperature: float = 1.0, top_k: int = None):
        """Probability distribution over the next token, given a context."""
        idx_cond = idx[:, -self.cfg.block_size:]          # never exceed the context window
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)  # only the last position matters
        if top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            kth = torch.topk(logits, k).values[:, [-1]]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        return softmax(logits, dim=-1)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """Autoregressive generation: predict, sample, append, repeat."""
        for _ in range(max_new_tokens):
            probs = self.next_token_probs(idx, temperature, top_k)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """Same as generate() but yields one token id at a time (for live typing in the UI)."""
        for _ in range(max_new_tokens):
            probs = self.next_token_probs(idx, temperature, top_k)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            yield next_id.item()

    def attention_maps(self):
        """List (one per layer) of the last attention weights, shape (B, n_head, T, T)."""
        return [block.attn.last_attention for block in self.blocks]
