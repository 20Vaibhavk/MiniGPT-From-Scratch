"""
MiniGPT Playground: a small web app to show the trained model in action.

    python app.py            ->  open http://127.0.0.1:7860

Tabs
  1. Generate          live, token-by-token text generation (like ChatGPT typing)
  2. Next-token lab    see the probability of every possible next character
  3. Attention viewer  heatmap of who-looks-at-whom inside any layer / head
  4. Architecture      the diagram and the exact configuration of this model
"""
import os

import gradio as gr
import torch
from matplotlib.figure import Figure

from minigpt import load_checkpoint  # noqa: E402

CKPT = os.environ.get("MINIGPT_CKPT", "checkpoints/minigpt.pt")
model, tokenizer, ckpt = load_checkpoint(CKPT)
cfg = model.cfg
ARCH_IMG = os.path.join("diagrams", "08_gpt_architecture.png")


def encode(text):
    ids = tokenizer.encode(text) or tokenizer.encode("\n")
    return torch.tensor([ids], dtype=torch.long)


# ------------------------------------------------------------------ 1. generate
def generate(prompt, max_tokens, temperature, top_k, seed):
    if seed >= 0:
        torch.manual_seed(int(seed))
    idx = encode(prompt)
    text = prompt
    for token_id in model.generate_stream(idx, int(max_tokens), float(temperature), int(top_k) or None):
        text += tokenizer.decode([token_id])
        yield text


# ------------------------------------------------------------------ 2. next-token lab
def show(ch):
    return {"\n": "\\n", " ": "space"}.get(ch, ch)


def next_token_plot(context, temperature, top_k):
    probs = model.next_token_probs(encode(context), float(temperature), int(top_k) or None)[0]
    top = torch.topk(probs, 12)
    labels = [show(tokenizer.decode([i])) for i in top.indices.tolist()]
    fig = Figure(figsize=(9, 4.8), dpi=110)
    ax = fig.subplots()
    bars = ax.barh(labels[::-1], top.values.tolist()[::-1], color="#2563eb")
    bars[-1].set_color("#ea580c")
    for b, v in zip(bars, top.values.tolist()[::-1]):
        ax.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2, f"{v:.1%}", va="center", fontsize=10)
    ax.set_xlim(0, max(top.values.tolist()) * 1.18)
    ax.set_title(f"What comes after ...{context[-25:]!r} ?", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def append_token(context, temperature, top_k, greedy):
    probs = model.next_token_probs(encode(context), float(temperature), int(top_k) or None)
    next_id = probs.argmax(-1).item() if greedy else torch.multinomial(probs, 1).item()
    context = context + tokenizer.decode([next_id])
    return context, next_token_plot(context, temperature, top_k)


# ------------------------------------------------------------------ 3. attention viewer
def attention_plot(text, layer, head):
    text = text[-48:]   # keep the heatmap readable
    model(encode(text))
    w = model.attention_maps()[int(layer)][0, int(head)].numpy()
    labels = [show(c) for c in text]
    fig = Figure(figsize=(8, 6.6), dpi=110)
    ax = fig.subplots()
    ax.imshow(w, cmap="Oranges")
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.xaxis.tick_top()
    ax.set_xlabel("...is looking at this token")
    ax.set_ylabel("this token...")
    ax.set_title(f"Layer {int(layer)}  |  Head {int(head)}", fontsize=13)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ UI
config_md = f"""
| setting | value |
|---|---|
| parameters | **{model.num_parameters():,}** |
| vocabulary (characters) | {cfg.vocab_size} |
| context length | {cfg.block_size} |
| transformer blocks | {cfg.n_layer} |
| attention heads per block | {cfg.n_head} |
| embedding size (d_model) | {cfg.n_embd} |
| positional encoding | {cfg.pos_encoding} |
| trained steps | {ckpt['step']} |
| validation loss | {ckpt['val_loss']:.3f} |
"""

with gr.Blocks(title="MiniGPT Playground") as demo:
    gr.Markdown("# MiniGPT Playground\nA GPT-style transformer built and trained from scratch.")

    with gr.Tab("1. Generate"):
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(label="Prompt", value="ROMEO:\n", lines=3)
                max_tokens = gr.Slider(20, 1000, value=300, step=10, label="Max new tokens")
                temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature (creativity)")
                top_k = gr.Slider(0, 65, value=40, step=1, label="Top-k (0 = off)")
                seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                go = gr.Button("Generate", variant="primary")
            with gr.Column(scale=2):
                output = gr.Textbox(label="MiniGPT writes...", lines=22)
        go.click(generate, [prompt, max_tokens, temperature, top_k, seed], output)
        gr.Examples([["ROMEO:\n"], ["JULIET:\nO "], ["KING HENRY:\nMy lords, "], ["To be, or not to be"]],
                    inputs=[prompt])

    with gr.Tab("2. Next-token lab"):
        gr.Markdown("The model only ever answers one question: *which character comes next?*")
        with gr.Row():
            with gr.Column(scale=1):
                ctx = gr.Textbox(label="Context", value="To be, or not to b", lines=4)
                t2 = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature")
                k2 = gr.Slider(0, 65, value=0, step=1, label="Top-k (0 = off)")
                with gr.Row():
                    show_btn = gr.Button("Show probabilities", variant="primary")
                    greedy_btn = gr.Button("Append most likely")
                    sample_btn = gr.Button("Append sampled")
            plot = gr.Plot(label="Next-token probabilities")
        show_btn.click(next_token_plot, [ctx, t2, k2], plot)
        greedy_btn.click(lambda c, t, k: append_token(c, t, k, True), [ctx, t2, k2], [ctx, plot])
        sample_btn.click(lambda c, t, k: append_token(c, t, k, False), [ctx, t2, k2], [ctx, plot])

    with gr.Tab("3. Attention viewer"):
        gr.Markdown("Each row shows where one character looks. The upper triangle is empty: "
                    "**the causal mask stops the model from seeing the future.**")
        with gr.Row():
            with gr.Column(scale=1):
                att_text = gr.Textbox(label="Text (last 48 characters are shown)",
                                      value="ROMEO:\nBut soft, what light through yonder window",
                                      lines=3)
                layer = gr.Slider(0, cfg.n_layer - 1, value=cfg.n_layer - 1, step=1, label="Layer")
                head = gr.Slider(0, cfg.n_head - 1, value=0, step=1, label="Head")
                att_btn = gr.Button("Show attention", variant="primary")
            att_plot = gr.Plot(label="Attention weights")
        att_btn.click(attention_plot, [att_text, layer, head], att_plot)
        layer.release(attention_plot, [att_text, layer, head], att_plot)
        head.release(attention_plot, [att_text, layer, head], att_plot)

    with gr.Tab("4. Architecture"):
        with gr.Row():
            if os.path.exists(ARCH_IMG):
                gr.Image(ARCH_IMG, show_label=False)
            gr.Markdown(config_md)

if __name__ == "__main__":
    demo.launch()
