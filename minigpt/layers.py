"""
Basic neural-network building blocks written from scratch.

We only use PyTorch for two things:
  1. Tensors (fast n-dimensional arrays that can live on CPU or GPU)
  2. Autograd (automatic computation of gradients for back-propagation)

Every layer below (Linear, Embedding, LayerNorm, GELU, Dropout, Softmax,
Cross-Entropy) is implemented by hand with plain tensor maths, so you can see
exactly what happens inside a transformer.
"""
import math

import torch
import torch.nn as nn


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
