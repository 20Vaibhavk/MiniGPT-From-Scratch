"""
Sanity tests: every hand-written layer must match PyTorch's built-in version,
and the model must respect causality (no peeking at future tokens).

    python -m pytest tests -q      (or)      python tests/test_minigpt.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from minigpt import BPETokenizer, CharTokenizer, GPTConfig, MiniGPT  # noqa: E402
from minigpt.layers import GELU, LayerNorm, cross_entropy, softmax  # noqa: E402
from minigpt.optim import AdamW  # noqa: E402

torch.manual_seed(0)


def small_model(**kw):
    cfg = GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0, **kw)
    return MiniGPT(cfg)


def test_softmax_matches_torch():
    x = torch.randn(4, 10)
    assert torch.allclose(softmax(x), torch.softmax(x, dim=-1), atol=1e-6)


def test_layernorm_matches_torch():
    x = torch.randn(3, 5, 8)
    assert torch.allclose(LayerNorm(8)(x), F.layer_norm(x, (8,)), atol=1e-5)


def test_gelu_matches_torch():
    x = torch.randn(100)
    assert torch.allclose(GELU()(x), F.gelu(x, approximate="tanh"), atol=1e-6)


def test_cross_entropy_matches_torch():
    logits, targets = torch.randn(7, 11), torch.randint(0, 11, (7,))
    assert torch.allclose(cross_entropy(logits, targets), F.cross_entropy(logits, targets), atol=1e-6)


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


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nall {len(tests)} tests passed")
