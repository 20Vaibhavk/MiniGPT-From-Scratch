"""MiniGPT - a GPT-style transformer built from scratch for learning."""
import torch

from .config import GPTConfig, TrainConfig
from .model import MiniGPT
from .tokenizer import BPETokenizer, CharTokenizer


def load_checkpoint(path: str, device: str = "cpu"):
    """Return (model, tokenizer, checkpoint_dict) from a file saved by train.py."""
    ckpt = torch.load(path, map_location=device)
    cfg = GPTConfig(**ckpt["model_config"])
    model = MiniGPT(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tokenizer = CharTokenizer.from_dict(ckpt["tokenizer"])
    return model, tokenizer, ckpt


__all__ = ["GPTConfig", "TrainConfig", "MiniGPT", "CharTokenizer", "BPETokenizer", "load_checkpoint"]
