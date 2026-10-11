"""
AdamW optimiser and learning-rate schedule, written from scratch.

Adam keeps two running averages per parameter:
  m = average gradient            (momentum: which way to go)
  v = average squared gradient    (how noisy that direction is)
and moves each weight by  lr * m / (sqrt(v) + eps).
The "W" means weight decay is applied directly to the weights (decoupled).
"""
import math

import torch


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            m_hat = m / (1 - self.beta1 ** self.t)      # bias correction
            v_hat = v / (1 - self.beta2 ** self.t)
            if p.dim() >= 2 and self.weight_decay > 0:   # decay matrices, not biases/norms
                p.mul_(1 - self.lr * self.weight_decay)
            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def state_dict(self):
        return {"t": self.t, "m": self.m, "v": self.v}


def clip_grad_norm(params, max_norm: float) -> float:
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.sqrt(sum((g ** 2).sum() for g in grads))
    if total > max_norm:
        for g in grads:
            g.mul_(max_norm / (total + 1e-6))
    return total.item()


def cosine_lr(step: int, max_iters: int, base_lr: float, min_lr: float, warmup: int) -> float:
    """Linear warm-up, then cosine decay from base_lr to min_lr."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, max_iters - warmup))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
