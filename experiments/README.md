# Experiments

## Ready to run

| Script | What it shows |
|---|---|
| `python experiments/top_p_sampling.py` | Greedy vs top-k vs top-p (nucleus) sampling on the same prompt. Greedy gets stuck repeating itself; top-p stays varied. |
| `python experiments/surprise_detector.py` | Uses the model as an anomaly detector: lines it finds unusual get a high loss. Try `--file your_lines.txt`. |

## Ideas to try next

Change one thing at a time and write down the validation loss printed by `train.py`.

| Idea | How |
|---|---|
| Bigger model | `python train.py --n-layer 6 --n-embd 256` |
| Longer memory | `python train.py --block-size 256` |
| Original position encoding | `python train.py --pos-encoding sinusoidal` and compare the loss |
| See overfitting live | `python train.py --dropout 0.0` and watch train and validation loss drift apart |
| Your own data | `python train.py --data my_text.txt --out-dir checkpoints/my_model`, then `python generate.py --ckpt checkpoints/my_model/minigpt.pt` |
| Hindi or any language | Character-level works with any script: train on a UTF-8 text file |
| Code completion | Concatenate your `.py` files into one text file and train on it |
| Log anomaly detection | Train on normal logs, then score new logs with `experiments/surprise_detector.py` |
| Weight tying | Share `token_embedding.weight` with the output head (GPT-2 does this) and compare parameters and loss |
| KV cache | Store past keys and values in `MultiHeadAttention` so generation does not recompute them |
| BPE training | Use `BPETokenizer` from `minigpt/tokenizer.py` instead of characters |
| GPU | The code uses CUDA automatically when available: train a much bigger model |
