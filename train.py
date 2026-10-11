"""
Train MiniGPT on any text file (default: Tiny Shakespeare, ~1 MB).

    python train.py                       # default settings, ~15-25 min on a laptop CPU
    python train.py --max-iters 300       # quick demo run for the video
    python train.py --data my_text.txt    # train on your own text

Outputs (in checkpoints/):
    minigpt.pt          model weights + config + tokenizer
    history.json        loss curve values
    loss_curve.png      plot of train / validation loss
    samples.txt         text generated at every sampling step
"""
import argparse
import json
import os
import time

import torch

from minigpt import CharTokenizer, GPTConfig, MiniGPT, TrainConfig
from minigpt.data import TextDataset, download_tiny_shakespeare, load_text
from minigpt.optim import AdamW, clip_grad_norm, cosine_lr


def parse_args():
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


def main():
    args = parse_args()
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


if __name__ == "__main__":
    main()
