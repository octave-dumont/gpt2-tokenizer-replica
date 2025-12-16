# Minimal GPT‑2‑Style Byte‑Level BPE Tokenizer in Python

Compact implementation of a GPT‑2‑style tokenizer:

- **GPT‑2 regex pre‑tokenization** using the GPT-4 improved pattern
- **Byte‑level base vocabulary** (UTF‑8 bytes `0–255`)
- **Byte‑Pair Encoding (BPE)** merge learning + merge application
- Simple `train` / `encode` / `decode` / `save` / `load` API

> This repo is meant for experimentation, it does not claim to be a production-ready library.

---

## Repository structure

All code and example artifacts live in `src/`:

- `tokenizer.py`: core implementation (`BPETokenizer`)
- `train_and_demo.py`: CLI demo (train + encode/decode + quick inspection output)
- `training_text.txt`: small sample training corpora
- `Tokenizer_example.txt`: example tokenizer saved with `tokenizer.save(...)`  
   > *(JSON content stored with a `.txt` extension, the extension doesn’t matter.)*

---

## Installation

**Python 3.10+** recommended.

```bash
pip install pydantic regex
```

---

## Quick start (CLI)

### Run from `src/` (uses defaults)

```bash
cd src
python train_and_demo.py
```

Defaults:
- training file: `training_text.txt`
- `vocab_size`: `500`
- verbose: `False`
- runs an encode-decode round‑trip demo
- prints a small sample of learned tokens and (optionally) basic training stats

### Run from repo root

```bash
python src/train_and_demo.py --train-file src/training_text.txt
```

### CLI arguments

```bash
python train_and_demo.py --train-file training_text.txt --vocab-size 800 --example-text "Hello BPE!"
```

- `--train-file`: path to a UTF‑8 text corpus
- `--vocab-size`: target vocabulary size (`> 256`)
- `--example-text`: text used for the round‑trip demo
- `--verbose`: prints training statistics

---

## Use as a module (Python)

### Train and save

```python
from pathlib import Path
from tokenizer import BPETokenizer, load_text

text = load_text(Path("training_text.txt"))
tok = BPETokenizer.train(text, vocab_size=500, verbose=True)

tok.save("Tokenizer_example.txt")  # JSON file (extension is arbitrary)
```

### Load + encode/decode

```python
from tokenizer import BPETokenizer

tok = BPETokenizer.load("Tokenizer_example.txt")

ids = tok.encode("Hello BPE world!")
print("IDs:", ids)
print("Decoded:", tok.decode(ids))
```

---

## High-level algorithm

1. **Pre-tokenize** text into pieces using a GPT‑2‑style regex.
2. Convert each piece to **UTF‑8 bytes** (initial token ids are `0–255`).
3. Repeatedly:
   - count adjacent token pairs in the corpus,
   - merge the most frequent pair into a new token id (`256+`),
   - store the merge rule.
4. For encoding, apply learned merges **within each piece**, then concatenate ids.

---

## Limitations

This project aims to be a personal attempt at the tokenization process:

- not optimized for large corpora (in‑memory training)
- no special tokens (<BOS>/<EOS>/<UNK>), no normalization pipeline
- merge learning is straightforward frequency‑based BPE

For a production tokenizer, consider **tiktoken** or **HuggingFace tokenizers**.

---

