"""
Configuration objects for MiniGPT.

Keeping every "knob" of the model and the training run in one place makes it
easy to experiment: change a number here, retrain, compare.
"""
from dataclasses import dataclass, asdict


@dataclass
class GPTConfig:
    vocab_size: int = 65        # how many different tokens exist (set from the tokenizer)
    block_size: int = 128       # context length: how many tokens the model can look at
    n_layer: int = 4            # how many transformer blocks are stacked
    n_head: int = 4             # attention heads inside every block
    n_embd: int = 192           # size of every token vector (called d_model in the paper)
    dropout: float = 0.1        # regularisation: randomly switch off 10% of activations
    pos_encoding: str = "learned"  # "learned" (GPT style) or "sinusoidal" (original paper)

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    batch_size: int = 32        # sequences processed in parallel per step
    max_iters: int = 2500       # total optimisation steps
    eval_interval: int = 250    # how often we measure train/val loss
    eval_iters: int = 40        # batches averaged for each loss estimate
    learning_rate: float = 1e-3
    min_lr: float = 1e-4        # cosine schedule decays to this value
    warmup_iters: int = 100     # linear warm-up steps at the start
    weight_decay: float = 0.1
    grad_clip: float = 1.0      # cap the gradient norm to keep training stable
    sample_every: int = 500     # print a generated sample during training
    seed: int = 1337

    def to_dict(self):
        return asdict(self)
