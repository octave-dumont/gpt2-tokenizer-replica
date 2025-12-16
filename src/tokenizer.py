from __future__ import annotations
import json
import time

import regex as re
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Dict, Tuple, Sequence, Union


Pair = Tuple[int, int]


TOKEN_PATTERN = re.compile(r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{2,}|"""
                           r"""[^\r\n\p{L}\p{N}]?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""", 
                           re.UNICODE)



def get_counts(tokens: Sequence[int]) -> Dict[Pair, int]:
    """
    Count the occurrences of each adjacent token pair in a single sequence.
    """
    num_occ: Dict[Pair, int] = {}
    for pair in zip(tokens, tokens[1:]):
        num_occ[pair] = num_occ.get(pair, 0) + 1
    return num_occ


def merge(tokens: Sequence[int], pair: Pair, new_rep: int) -> List[int]:
    """
    Return a new list where every occurrence of `pair` in `tokens`
    is replaced by a single new token `new_rep`.
    """
    merged_tokens: List[int] = []
    i = 0
    n = len(tokens)
    while i < n:
        if i < n - 1 and (tokens[i], tokens[i + 1]) == pair:
            merged_tokens.append(new_rep)
            i += 2
        else:
            merged_tokens.append(tokens[i])
            i += 1
    return merged_tokens


def merge_all(segments: Sequence[Sequence[int]], num_merges: int) -> Tuple[List[List[int]], Dict[int, Pair]]:
    """
    Perform `num_merges` BPE merges over all regex-pre-tokenized segments.
    Returns:
      - the merged token sequences as a list of lists of tokens
      - the history as a dict: `{new_token_id: (token_a, token_b)}`

    Merges are learned and applied within each segment only:
    we never consider pairs that cross segment boundaries.
    """
    segs: List[List[int]] = [list(seg) for seg in segments]
    merges_history: Dict[int, Pair] = {}

    for i in range(num_merges):
        counts: Dict[Pair, int] = {}
        for seg in segs:
            seg_counts = get_counts(seg)
            for pair, c in seg_counts.items():
                counts[pair] = counts.get(pair, 0) + c

        if not counts:
            break

        top_pair = max(counts, key=counts.get)
        new_rep = 256 + i
        
        segs = [merge(seg, top_pair, new_rep) for seg in segs]
        merges_history[new_rep] = top_pair

    return segs, merges_history



class BPETokenizer(BaseModel):
    """
    Byte-pair tokenizer trained from raw text.
    """
    vocab_size: int = Field(..., gt=256, description="Total vocabulary size (must be > 256).")
    vocab: Dict[int, bytes]  # token_id -> byte sequence
    merges: Dict[Pair, int]  # (token_a, token_b) -> new_token_id


    @classmethod
    def train(cls, text: str, vocab_size: int, verbose: bool = False) -> "BPETokenizer":
        """
        Train a tokenizer from raw text and target vocab size.
        """
        segments: List[List[int]] = []
        for subt in TOKEN_PATTERN.findall(text):
            if not subt:
                continue
            segments.append(list(subt.encode("utf-8")))

        if not segments:
            raise ValueError("Training text must be non-empty.")
        
        start_time = time.time()     

        num_merges = vocab_size - 256
        merged_segments, merges_history = merge_all(segments, num_merges)

        # Base vocabulary: raw bytes 0–255
        vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        for new_token, pair in merges_history.items():
            vocab[new_token] = vocab[pair[0]] + vocab[pair[1]]

        merges: Dict[Pair, int] = {pair: new_token for new_token, pair in merges_history.items()}

        if verbose and merged_segments:
            duration = time.time() - start_time
            base_len = sum(len(seg) for seg in segments)
            merged_len = max(sum(len(seg) for seg in merged_segments), 1)
            compression_ratio = base_len / merged_len
            print(
                f"[BPETokenizer] Trained with vocab_size={vocab_size}, "
                f"compression_ratio={compression_ratio:.4f}\n",
                f"Training duration: {duration:.4f}s"
            )

        return cls(vocab_size=len(vocab), vocab=vocab, merges=merges)
    
    def _bpe_on_bytes(self, byte_ids: List[int]) -> List[int]:
        """
        Apply BPE merges to a sequence of byte ids (one pre-token).
        """
        tokens: List[int] = list(byte_ids)
        while len(tokens) > 1:
            stats = get_counts(tokens)
            if not stats:
                break

            candidate_pairs = {p: self.merges[p] for p in stats if p in self.merges}
            if not candidate_pairs:
                break

            pair = min(candidate_pairs, key=candidate_pairs.get)
            tokens = merge(tokens, pair, self.merges[pair])
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs using learned merges, with regex pre-tokenization.
        """
        all_ids: List[int] = []

        for subt in TOKEN_PATTERN.findall(text):
            if not subt:
                continue
            byte_ids = list(subt.encode("utf-8"))
            piece_ids = self._bpe_on_bytes(byte_ids)
            all_ids.extend(piece_ids)

        return all_ids


    def decode(self, ids: Sequence[int]) -> str:
        """
        Decode token IDs back to UTF-8 text.
        """
        try:
            return b"".join(self.vocab[i] for i in ids).decode("utf-8")
        except KeyError as e:
            raise ValueError(f"Unknown token id in decode: {e.args[0]}") from e

    def to_dict(self) -> dict:
        """
        Serialize the tokenizer to a JSON-serializable dict.
        """
        return {
            "vocab_size": self.vocab_size,
            "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
            "merges": {f"{a},{b}": t for (a, b), t in self.merges.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BPETokenizer":
        """
        Reconstruct a tokenizer instance from a dict produced by `to_dict`.
        """
        vocab: Dict[int, bytes] = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
        merges: Dict[Pair, int] = {}
        for key, token_id in data["merges"].items():
            a_str, b_str = key.split(",")
            merges[(int(a_str), int(b_str))] = token_id
        return cls(vocab_size=data["vocab_size"], vocab=vocab, merges=merges)

    def save(self, path: Union[Path, str]) -> None:
        """
        Save the tokenizer to a JSON file.
        """
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[Path, str]) -> "BPETokenizer":
        """
        Load a tokenizer from a JSON file.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def load_text(path: Path) -> str:
    """
    Load text from a UTF-8 encoded file.
    """
    return path.read_text(encoding="utf-8")

__all__ = ["BPETokenizer", "load_text"]