"""
MiniGPT: a GPT-style transformer language model, written from scratch, in ONE file.
Companion code for the course "Build Your Own ChatGPT: LLM & Transformers From Scratch".

PyTorch is used only for tensors, autograd and nn.Module. Every layer, the attention mechanism,
the transformer, the AdamW optimiser and both tokenizers are written by hand below.

Usage
    python minigpt_complete.py train                      # train on Tiny Shakespeare (~30 min on a laptop CPU)
    python minigpt_complete.py train --max-iters 300      # quick demo run
    python minigpt_complete.py train --data my_text.txt   # train on your own text
    python minigpt_complete.py generate --prompt "ROMEO:" --tokens 400 --temperature 0.8
    python minigpt_complete.py test                       # 12 correctness checks

Contents
    1. Configuration          5. Tokenizers (character + BPE)
    2. Layers by hand         6. Data pipeline
    3. Attention              7. AdamW optimiser + schedule
    4. The MiniGPT model      8. Training, 9. Generation, 10. Tests, 11. Command line
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn



# ====================================================================================================
# 1. CONFIGURATION
# ====================================================================================================

# Configuration objects for MiniGPT.
#
# Keeping every "knob" of the model and the training run in one place makes it
# easy to experiment: change a number here, retrain, compare.

@dataclass
class GPTConfig:
    vocab_size: int = 65        # how many different tokens exist (set from the tokenizer)
    block_size: int = 128       # context length: how many tokens the model can look at
    n_layer: int = 4            # how many transformer blocks are stacked
    n_head: int = 4             # attention heads inside every block
    n_embd: int = 192           # size of every token vector (called d_model in the paper)
    dropout: float = 0.1        # regularisation: randomly switch off 10% of activations
    pos_encoding: str = "learned"  # "learned" (GPT style) or "sinusoidal" (original paper)

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    batch_size: int = 32        # sequences processed in parallel per step
    max_iters: int = 2500       # total optimisation steps
    eval_interval: int = 250    # how often we measure train/val loss
    eval_iters: int = 40        # batches averaged for each loss estimate
    learning_rate: float = 1e-3
    min_lr: float = 1e-4        # cosine schedule decays to this value
    warmup_iters: int = 100     # linear warm-up steps at the start
    weight_decay: float = 0.1
    grad_clip: float = 1.0      # cap the gradient norm to keep training stable
    sample_every: int = 500     # print a generated sample during training
    seed: int = 1337

    def to_dict(self):
        return asdict(self)


# ====================================================================================================
# 2. LAYERS BY HAND
# ====================================================================================================

# Basic neural-network building blocks written from scratch.
#
# We only use PyTorch for two things:
#   1. Tensors (fast n-dimensional arrays that can live on CPU or GPU)
#   2. Autograd (automatic computation of gradients for back-propagation)
#
# Every layer below (Linear, Embedding, LayerNorm, GELU, Dropout, Softmax,
# Cross-Entropy) is implemented by hand with plain tensor maths, so you can see
# exactly what happens inside a transformer.

class Linear(nn.Module):
    """y = x @ W + b  -- the most important operation in deep learning."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, std: float = 0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features) * std)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


class Embedding(nn.Module):
    """A lookup table: token id -> learnable vector."""

    def __init__(self, num_embeddings: int, embedding_dim: int, std: float = 0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * std)

    def forward(self, idx):
        # idx has shape (B, T) of integers; indexing returns (B, T, embedding_dim)
        return self.weight[idx]


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sin/cos position signal from 'Attention Is All You Need' (2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, block_size: int, d_model: int):
        super().__init__()
        pe = torch.zeros(block_size, d_model)
        position = torch.arange(block_size).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # not trained, but saved with the model

    def forward(self, positions):
        return self.pe[positions]


class LayerNorm(nn.Module):
    """Normalise every token vector to mean 0 / variance 1, then rescale.

    Keeps the numbers flowing through a deep network in a healthy range.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))   # learnable scale
        self.beta = nn.Parameter(torch.zeros(dim))   # learnable shift

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class GELU(nn.Module):
    """Smooth activation used by GPT-2/3 (tanh approximation)."""

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class Dropout(nn.Module):
    """During training, randomly zero a fraction p of values and rescale the rest."""

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = (torch.rand_like(x) > self.p).to(x.dtype)
        return x * keep / (1.0 - self.p)


def softmax(x, dim: int = -1):
    """Turn any list of scores into probabilities that sum to 1.

    Subtracting the max first is a numerical-stability trick: it does not
    change the result but prevents exp() from overflowing.
    """
    x = x - x.max(dim=dim, keepdim=True).values
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)


def log_softmax(x, dim: int = -1):
    m = x.max(dim=dim, keepdim=True).values
    logsumexp = m + torch.log(torch.exp(x - m).sum(dim=dim, keepdim=True))
    return x - logsumexp


def cross_entropy(logits, targets):
    """Average negative log-probability that the model gave to the correct token.

    logits:  (N, vocab_size) raw scores
    targets: (N,) correct token ids
    """
    log_probs = log_softmax(logits, dim=-1)
    correct = log_probs[torch.arange(targets.shape[0]), targets]
    return -correct.mean()


# ====================================================================================================
# 3. ATTENTION
# ====================================================================================================

# Attention: the heart of the transformer.
#
#     Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
#
# Every token asks a question (Query), every token advertises what it contains
# (Key), and every token offers information (Value). The dot product Q.K says
# "how relevant is that token to me?", softmax turns the scores into weights,
# and the output is a weighted mix of the Values.

def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """q, k, v: (..., T, head_size). mask: 1 = allowed, 0 = blocked."""
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)          # (..., T, T)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))   # hide the future
    weights = softmax(scores, dim=-1)                            # each row sums to 1
    if dropout is not None:
        weights = dropout(weights)
    return weights @ v, weights


class SelfAttentionHead(nn.Module):
    """One head of causal self-attention. Used in the lecture to explain the idea."""

    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        self.query = Linear(n_embd, head_size, bias=False)
        self.key = Linear(n_embd, head_size, bias=False)
        self.value = Linear(n_embd, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)
        out, _ = scaled_dot_product_attention(q, k, v, self.mask[:T, :T])
        return out


class MultiHeadAttention(nn.Module):
    """Several attention heads running in parallel, each in its own sub-space.

    Implemented efficiently: one Linear produces Q, K and V for all heads at
    once, then we reshape so that heads become an extra batch dimension.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.qkv = Linear(n_embd, 3 * n_embd)
        self.proj = Linear(n_embd, n_embd)
        self.attn_dropout = Dropout(dropout)
        self.resid_dropout = Dropout(dropout)
        # Lower-triangular causal mask: token t may only look at tokens 0..t
        mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        self.register_buffer("mask", mask)
        self.last_attention = None  # stored for visualisation

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)                      # each (B, T, C)
        # (B, T, C) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        out, weights = scaled_dot_product_attention(
            q, k, v, self.mask[:, :, :T, :T], self.attn_dropout
        )
        self.last_attention = weights.detach()                       # (B, n_head, T, T)

        # put the heads back side by side: (B, n_head, T, hs) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(out))


# ====================================================================================================
# 4. THE MINIGPT MODEL
# ====================================================================================================

# MiniGPT: a decoder-only transformer, the same family as GPT-2, GPT-3 and ChatGPT.
#
#     token ids
#       -> token embedding + position embedding
#       -> N x [ LayerNorm -> Multi-Head Attention -> +residual
#                LayerNorm -> Feed-Forward          -> +residual ]
#       -> final LayerNorm
#       -> Linear "language-model head"  -> logits over the vocabulary

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


# ====================================================================================================
# 5. TOKENIZERS
# ====================================================================================================

# Tokenizers turn text into integers and back.
#
# CharTokenizer  - one token per character. Tiny vocabulary, perfect for learning.
# BPETokenizer   - Byte-Pair Encoding, the algorithm behind GPT-2/3/4 tokenizers,
#                  written from scratch so you can see how "sub-words" are born.

class CharTokenizer:
    def __init__(self, chars=None):
        self.chars = sorted(set(chars)) if chars else []
        self._build()

    def _build(self):
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}   # string -> integer
        self.itos = {i: ch for i, ch in enumerate(self.chars)}   # integer -> string

    @classmethod
    def from_text(cls, text: str):
        return cls(text)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str):
        # characters never seen in training are skipped
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def to_dict(self):
        return {"type": "char", "chars": self.chars}

    @classmethod
    def from_dict(cls, d):
        tok = cls()
        tok.chars = list(d["chars"])
        tok._build()
        return tok

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class BPETokenizer:
    """Minimal byte-level Byte-Pair Encoding.

    Start from the 256 possible bytes. Repeatedly find the most frequent pair
    of neighbouring tokens and merge it into a brand-new token. After N merges
    the vocabulary has 256 + N tokens, and common chunks like "the" or "ing"
    become single tokens.
    """

    def __init__(self):
        self.merges = {}                                   # (a, b) -> new id
        self.vocab = {i: bytes([i]) for i in range(256)}   # id -> bytes

    @staticmethod
    def _pair_counts(ids):
        return Counter(zip(ids, ids[1:]))

    @staticmethod
    def _merge(ids, pair, new_id):
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text: str, vocab_size: int = 300, verbose: bool = False):
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        for step in range(vocab_size - 256):
            counts = self._pair_counts(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = 256 + step
            ids = self._merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {step + 1:3d}: {self.vocab[new_id]!r:<14} (seen {counts[pair]} times)")
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = self._pair_counts(ids)
            # apply the merge that was learned earliest
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_strings(self, ids):
        return [self.vocab[i].decode("utf-8", errors="replace") for i in ids]


# ====================================================================================================
# 6. DATA PIPELINE
# ====================================================================================================

# Data pipeline: download text, tokenize, split, and serve random batches.
#
# A language model learns one simple game: given the tokens so far, predict
# the next one. So for a chunk  "Hello"  the training pairs are:
#     input  H        -> target e
#     input  He       -> target l
#     input  Hel      -> target l
#     input  Hell     -> target o
# One chunk of length T gives T training examples at once.

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(path: str = "data/tinyshakespeare.txt") -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"Downloading Tiny Shakespeare -> {path}")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    return path


def load_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


class TextDataset:
    def __init__(self, text: str, tokenizer, train_split: float = 0.9):
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(train_split * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str = "cpu"):
        data = self.train_data if split == "train" else self.val_data
        starts = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in starts])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])   # shifted by one
        return x.to(device), y.to(device)


# ====================================================================================================
# 7. ADAMW OPTIMISER AND LEARNING-RATE SCHEDULE
# ====================================================================================================

# AdamW optimiser and learning-rate schedule, written from scratch.
#
# Adam keeps two running averages per parameter:
#   m = average gradient            (momentum: which way to go)
#   v = average squared gradient    (how noisy that direction is)
# and moves each weight by  lr * m / (sqrt(v) + eps).
# The "W" means weight decay is applied directly to the weights (decoupled).

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            m_hat = m / (1 - self.beta1 ** self.t)      # bias correction
            v_hat = v / (1 - self.beta2 ** self.t)
            if p.dim() >= 2 and self.weight_decay > 0:   # decay matrices, not biases/norms
                p.mul_(1 - self.lr * self.weight_decay)
            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def state_dict(self):
        return {"t": self.t, "m": self.m, "v": self.v}


def clip_grad_norm(params, max_norm: float) -> float:
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.sqrt(sum((g ** 2).sum() for g in grads))
    if total > max_norm:
        for g in grads:
            g.mul_(max_norm / (total + 1e-6))
    return total.item()


def cosine_lr(step: int, max_iters: int, base_lr: float, min_lr: float, warmup: int) -> float:
    """Linear warm-up, then cosine decay from base_lr to min_lr."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, max_iters - warmup))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ====================================================================================================
# 8. TRAINING
# ====================================================================================================

def parse_train_args():
    g, t = GPTConfig(), TrainConfig()
    p = argparse.ArgumentParser(description="Train MiniGPT from scratch")
    p.add_argument("--data", default=None, help="path to a .txt file (default: Tiny Shakespeare)")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--block-size", type=int, default=g.block_size)
    p.add_argument("--n-layer", type=int, default=g.n_layer)
    p.add_argument("--n-head", type=int, default=g.n_head)
    p.add_argument("--n-embd", type=int, default=g.n_embd)
    p.add_argument("--dropout", type=float, default=g.dropout)
    p.add_argument("--pos-encoding", choices=["learned", "sinusoidal"], default=g.pos_encoding)
    p.add_argument("--batch-size", type=int, default=t.batch_size)
    p.add_argument("--max-iters", type=int, default=t.max_iters)
    p.add_argument("--eval-interval", type=int, default=t.eval_interval)
    p.add_argument("--eval-iters", type=int, default=t.eval_iters)
    p.add_argument("--lr", type=float, default=t.learning_rate)
    p.add_argument("--sample-every", type=int, default=t.sample_every)
    p.add_argument("--seed", type=int, default=t.seed)
    return p.parse_args()


@torch.no_grad()
def estimate_loss(model, dataset, cfg, tcfg, device):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(tcfg.eval_iters)
        for i in range(tcfg.eval_iters):
            x, y = dataset.get_batch(split, tcfg.batch_size, cfg.block_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def sample(model, tokenizer, device, prompt="\n", n=200):
    model.eval()
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    text = tokenizer.decode(model.generate(idx, n, temperature=0.8, top_k=40)[0].tolist())
    model.train()
    return text


def save_loss_plot(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.plot(history["step"], history["train"], label="train loss", color="#2563eb", lw=2.5)
    ax.plot(history["step"], history["val"], label="validation loss", color="#f97316", lw=2.5)
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("MiniGPT learning curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train_main():
    args = parse_train_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. data + tokenizer
    path = args.data or download_tiny_shakespeare()
    text = load_text(path)
    tokenizer = CharTokenizer.from_text(text)
    dataset = TextDataset(text, tokenizer)
    print(f"text: {len(text):,} characters | vocab size: {tokenizer.vocab_size}")
    print(f"train tokens: {len(dataset.train_data):,} | val tokens: {len(dataset.val_data):,}")

    # 2. model
    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
        n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout, pos_encoding=args.pos_encoding,
    )
    tcfg = TrainConfig(
        batch_size=args.batch_size, max_iters=args.max_iters, eval_interval=args.eval_interval,
        eval_iters=args.eval_iters, learning_rate=args.lr, sample_every=args.sample_every, seed=args.seed,
    )
    model = MiniGPT(cfg).to(device)
    print(f"model parameters: {model.num_parameters():,} | device: {device}")

    # 3. optimiser
    optimizer = AdamW(model.parameters(), lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)

    history = {"step": [], "train": [], "val": [], "lr": []}
    samples_log = open(os.path.join(args.out_dir, "samples.txt"), "w", encoding="utf-8")
    best_val = float("inf")
    start = time.time()

    # 4. the training loop
    for step in range(tcfg.max_iters + 1):
        lr = cosine_lr(step, tcfg.max_iters, tcfg.learning_rate, tcfg.min_lr, tcfg.warmup_iters)
        optimizer.lr = lr

        if step % tcfg.eval_interval == 0 or step == tcfg.max_iters:
            losses = estimate_loss(model, dataset, cfg, tcfg, device)
            history["step"].append(step)
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])
            history["lr"].append(lr)
            elapsed = time.time() - start
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| lr {lr:.2e} | {elapsed:6.0f}s")
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "model_state": model.state_dict(),
                    "model_config": cfg.to_dict(),
                    "train_config": tcfg.to_dict(),
                    "tokenizer": tokenizer.to_dict(),
                    "step": step,
                    "val_loss": best_val,
                }, os.path.join(args.out_dir, "minigpt.pt"))

        if tcfg.sample_every and step % tcfg.sample_every == 0:
            s = sample(model, tokenizer, device)
            print(f"----- sample at step {step} -----\n{s}\n--------------------------------")
            samples_log.write(f"===== step {step} =====\n{s}\n\n")
            samples_log.flush()

        if step == tcfg.max_iters:
            break

        x, y = dataset.get_batch("train", tcfg.batch_size, cfg.block_size, device)
        _, loss = model(x, y)              # forward pass
        optimizer.zero_grad()
        loss.backward()                    # back-propagation
        clip_grad_norm(model.parameters(), tcfg.grad_clip)
        optimizer.step()                   # update the weights

    samples_log.close()
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    save_loss_plot(history, os.path.join(args.out_dir, "loss_curve.png"))
    print(f"done in {time.time() - start:.0f}s | best val loss {best_val:.4f} "
          f"| saved to {args.out_dir}/minigpt.pt")


# ====================================================================================================
# 9. GENERATION
# ====================================================================================================

def load_checkpoint(path: str, device: str = "cpu"):
    """Return (model, tokenizer, checkpoint_dict) from a file saved by train.py."""
    ckpt = torch.load(path, map_location=device)
    cfg = GPTConfig(**ckpt["model_config"])
    model = MiniGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tokenizer = CharTokenizer.from_dict(ckpt["tokenizer"])
    return model, tokenizer, ckpt


def generate_main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/minigpt.pt")
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    model, tokenizer, ckpt = load_checkpoint(args.ckpt)
    print(f"[loaded model from step {ckpt['step']} | val loss {ckpt['val_loss']:.3f}]\n")

    idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long)
    print(args.prompt, end="", flush=True)
    for token_id in model.generate_stream(idx, args.tokens, args.temperature, args.top_k):
        print(tokenizer.decode([token_id]), end="", flush=True)   # live "typing" effect
    print()


# ====================================================================================================
# 10. TESTS (run: python minigpt_complete.py test)
# ====================================================================================================

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))




def small_model(**kw):
    cfg = GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0, **kw)
    return MiniGPT(cfg)


def test_softmax_matches_torch():
    x = torch.randn(4, 10)
    assert torch.allclose(softmax(x), torch.softmax(x, dim=-1), atol=1e-6)


def test_layernorm_matches_torch():
    x = torch.randn(3, 5, 8)
    assert torch.allclose(LayerNorm(8)(x), torch.nn.functional.layer_norm(x, (8,)), atol=1e-5)


def test_gelu_matches_torch():
    x = torch.randn(100)
    assert torch.allclose(GELU()(x), torch.nn.functional.gelu(x, approximate="tanh"), atol=1e-6)


def test_cross_entropy_matches_torch():
    logits, targets = torch.randn(7, 11), torch.randint(0, 11, (7,))
    assert torch.allclose(cross_entropy(logits, targets), torch.nn.functional.cross_entropy(logits, targets), atol=1e-6)


def test_output_shapes():
    model = small_model()
    x = torch.randint(0, 20, (3, 16))
    logits, loss = model(x, x)
    assert logits.shape == (3, 16, 20)
    assert loss.dim() == 0


def test_causal_mask_no_future_leak():
    """Changing a future token must not change predictions for earlier positions."""
    model = small_model().eval()
    x = torch.randint(0, 20, (1, 10))
    x2 = x.clone()
    x2[0, 7] = (x2[0, 7] + 1) % 20
    a, _ = model(x)
    b, _ = model(x2)
    assert torch.allclose(a[0, :7], b[0, :7], atol=1e-6)
    assert not torch.allclose(a[0, 7:], b[0, 7:])


def test_attention_rows_sum_to_one():
    model = small_model().eval()
    model(torch.randint(0, 20, (2, 12)))
    for w in model.attention_maps():
        assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)
        assert torch.all(w.triu(diagonal=1) == 0)


def test_sinusoidal_variant_runs():
    model = small_model(pos_encoding="sinusoidal")
    logits, _ = model(torch.randint(0, 20, (1, 8)))
    assert logits.shape == (1, 8, 20)


def test_model_can_overfit_one_batch():
    model = small_model()
    opt = AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    x = torch.randint(0, 20, (4, 16))
    y = torch.randint(0, 20, (4, 16))
    for _ in range(200):
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.2


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer.from_text("hello world")
    assert tok.decode(tok.encode("hello")) == "hello"


def test_bpe_roundtrip_and_compression():
    text = "the cat sat on the mat. the cat ate the rat. " * 20
    bpe = BPETokenizer().train(text, vocab_size=280)
    ids = bpe.encode("the cat sat on the mat.")
    assert bpe.decode(ids) == "the cat sat on the mat."
    assert len(ids) < len("the cat sat on the mat.".encode())


def test_generate_length():
    model = small_model().eval()
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=25)
    assert out.shape == (1, 26)   # works beyond block_size thanks to cropping



def run_tests():
    torch.manual_seed(0)
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nall {len(tests)} tests passed")


# ====================================================================================================
# 11. COMMAND LINE
# ====================================================================================================

if __name__ == "__main__":
    commands = {"train": train_main, "generate": generate_main, "test": run_tests}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("usage: python minigpt_complete.py {train,generate,test} [options]")
        sys.exit(1)
    command = sys.argv.pop(1)
    commands[command]()
