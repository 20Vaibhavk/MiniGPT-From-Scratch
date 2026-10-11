"""
Experiment: nucleus (top-p) sampling, compared with greedy and top-k decoding.

    python experiments/top_p_sampling.py --prompt "JULIET:" --p 0.9

Top-p keeps the smallest set of tokens whose probabilities add up to p, so the number of
candidates grows when the model is unsure and shrinks when it is confident.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from minigpt import load_checkpoint  # noqa: E402
from minigpt.layers import softmax  # noqa: E402


@torch.no_grad()
def generate_top_p(model, idx, max_new_tokens, p=0.9, temperature=1.0):
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.cfg.block_size:])
        probs = softmax(logits[:, -1, :] / temperature, dim=-1)
        sorted_probs, sorted_ids = torch.sort(probs, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        # drop a token once the tokens before it already cover p
        sorted_probs[cumulative - sorted_probs > p] = 0.0
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(sorted_probs, 1)
        idx = torch.cat([idx, sorted_ids.gather(-1, choice)], dim=1)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/minigpt.pt")
    ap.add_argument("--prompt", default="JULIET:\n")
    ap.add_argument("--tokens", type=int, default=250)
    ap.add_argument("--p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()

    model, tok, _ = load_checkpoint(a.ckpt)
    start = torch.tensor([tok.encode(a.prompt)])

    runs = {
        "greedy (always the top token)": lambda: model.generate(start, a.tokens, temperature=1e-4),
        "top-k = 10": lambda: model.generate(start, a.tokens, temperature=1.0, top_k=10),
        f"top-p = {a.p}": lambda: generate_top_p(model, start, a.tokens, p=a.p),
    }
    for name, run in runs.items():
        torch.manual_seed(a.seed)
        print(f"\n===== {name} =====")
        print(tok.decode(run()[0].tolist()))


if __name__ == "__main__":
    main()
