"""
Tokenizers turn text into integers and back.

CharTokenizer  - one token per character. Tiny vocabulary, perfect for learning.
BPETokenizer   - Byte-Pair Encoding, the algorithm behind GPT-2/3/4 tokenizers,
                 written from scratch so you can see how "sub-words" are born.
"""
import json
from collections import Counter


class CharTokenizer:
    def __init__(self, chars=None):
        self.chars = sorted(set(chars)) if chars else []
        self._build()

    def _build(self):
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}   # string -> integer
        self.itos = {i: ch for i, ch in enumerate(self.chars)}   # integer -> string

    @classmethod
    def from_text(cls, text: str):
        return cls(text)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str):
        # characters never seen in training are skipped
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def to_dict(self):
        return {"type": "char", "chars": self.chars}

    @classmethod
    def from_dict(cls, d):
        tok = cls()
        tok.chars = list(d["chars"])
        tok._build()
        return tok

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class BPETokenizer:
    """Minimal byte-level Byte-Pair Encoding.

    Start from the 256 possible bytes. Repeatedly find the most frequent pair
    of neighbouring tokens and merge it into a brand-new token. After N merges
    the vocabulary has 256 + N tokens, and common chunks like "the" or "ing"
    become single tokens.
    """

    def __init__(self):
        self.merges = {}                                   # (a, b) -> new id
        self.vocab = {i: bytes([i]) for i in range(256)}   # id -> bytes

    @staticmethod
    def _pair_counts(ids):
        return Counter(zip(ids, ids[1:]))

    @staticmethod
    def _merge(ids, pair, new_id):
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text: str, vocab_size: int = 300, verbose: bool = False):
        assert vocab_size >= 256
        ids = list(text.encode("utf-8"))
        for step in range(vocab_size - 256):
            counts = self._pair_counts(ids)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            new_id = 256 + step
            ids = self._merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {step + 1:3d}: {self.vocab[new_id]!r:<14} (seen {counts[pair]} times)")
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = self._pair_counts(ids)
            # apply the merge that was learned earliest
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def token_strings(self, ids):
        return [self.vocab[i].decode("utf-8", errors="replace") for i in ids]
