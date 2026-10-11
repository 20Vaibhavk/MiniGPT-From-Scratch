"""
Data pipeline: download text, tokenize, split, and serve random batches.

A language model learns one simple game: given the tokens so far, predict
the next one. So for a chunk  "Hello"  the training pairs are:
    input  H        -> target e
    input  He       -> target l
    input  Hel      -> target l
    input  Hell     -> target o
One chunk of length T gives T training examples at once.
"""
import os
import urllib.request

import torch

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
