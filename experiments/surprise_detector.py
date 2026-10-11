"""
Experiment: use the language model as an anomaly detector.

A model that has learned what "normal" text looks like gives unusual text a high loss.
This script scores every line of a file and lists the most surprising ones.

    python experiments/surprise_detector.py                      # demo lines
    python experiments/surprise_detector.py --file my_lines.txt  # your own file

Train the model on normal data first (for example server logs: python train.py --data logs.txt),
then score new logs: lines with the highest loss are worth a look.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from minigpt import load_checkpoint  # noqa: E402
from minigpt.layers import cross_entropy  # noqa: E402

DEMO = [
    "ROMEO:\nBut soft, what light through yonder window breaks?",
    "KING HENRY:\nOnce more unto the breach, dear friends, once more;",
    "JULIET:\nO Romeo, Romeo, wherefore art thou Romeo?",
    "SERVER:\nERROR 503 upstream timeout at 02:13:44 node-7",
    "MACBETH:\nTomorrow, and tomorrow, and tomorrow,",
    "lol ok see u l8r, gonna grab pizza brb",
]


@torch.no_grad()
def surprise(model, tok, text):
    """Average loss per character: how unexpected this text is for the model."""
    ids = torch.tensor([tok.encode(text)[-(model.cfg.block_size + 1):]])
    if ids.shape[1] < 2:
        return float("nan")
    logits, _ = model(ids[:, :-1])
    return cross_entropy(logits[0], ids[0, 1:]).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/minigpt.pt")
    ap.add_argument("--file", default=None, help="text file, one item per line")
    a = ap.parse_args()

    model, tok, _ = load_checkpoint(a.ckpt)
    items = DEMO if a.file is None else [l for l in open(a.file, encoding="utf-8").read().split("\n") if l.strip()]
    scored = sorted(((surprise(model, tok, t), t) for t in items), reverse=True)
    print(f"{'loss':>6}  text (most surprising first)")
    for loss, text in scored:
        print(f"{loss:6.2f}  {text.replace(chr(10), ' / ')[:90]}")


if __name__ == "__main__":
    main()
