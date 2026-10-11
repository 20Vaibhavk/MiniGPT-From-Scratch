"""
Generate text with a trained MiniGPT.

    python generate.py --prompt "ROMEO:" --tokens 400 --temperature 0.8 --top-k 40
"""
import argparse

import torch

from minigpt import load_checkpoint


def main():
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


if __name__ == "__main__":
    main()
