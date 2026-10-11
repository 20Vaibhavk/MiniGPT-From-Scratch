"""
Attention: the heart of the transformer.

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

Every token asks a question (Query), every token advertises what it contains
(Key), and every token offers information (Value). The dot product Q.K says
"how relevant is that token to me?", softmax turns the scores into weights,
and the output is a weighted mix of the Values.
"""
import math

import torch
import torch.nn as nn

from .layers import Dropout, Linear, softmax


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
