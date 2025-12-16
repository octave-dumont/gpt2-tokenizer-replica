"""
Minimal byte-pair encoding (BPE) tokenizer implementation with Pydantic-based config.
"""

from pathlib import Path
import argparse
import sys
from random import sample

from pydantic import ValidationError

from tokenizer import BPETokenizer, load_text


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training and demo."""

    parser = argparse.ArgumentParser(
        description="Train a simple BPE tokenizer and run a small demo."
    )

    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("training_text.txt"),
        help="Text file used to train the tokenizer (default: training_text.txt).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=500,
        help="Target vocabulary size (> 256, default: 500).",
    )
    parser.add_argument(
        "--example-text",
        type=str,
        default="Here is an example text to encode-decode.",
        help="Example text to encode/decode for the demo.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print training statistics (compression ratio, time, etc.).",
    )

    return parser.parse_args()


def training_step(args: argparse.Namespace) -> BPETokenizer:
    """Load training text and train a BPE tokenizer."""
    try:
        text = load_text(args.train_file)
    except FileNotFoundError:
        print(f"[ERROR] Training file not found: {args.train_file}", file=sys.stderr)
        raise SystemExit(1)

    if not text:
        print(f"[ERROR] Training file is empty: {args.train_file}", file=sys.stderr)
        raise SystemExit(1)

    try:
        tokenizer = BPETokenizer.train(text, vocab_size=args.vocab_size, verbose=getattr(args, "verbose", True))
    except (ValueError, ValidationError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(1)
    
    return tokenizer


def demo(args: argparse.Namespace, tokenizer: BPETokenizer) -> None:
    """Run a simple encode/decode demo using the trained tokenizer."""
    example = args.example_text
    ids = tokenizer.encode(example)
    decoded = tokenizer.decode(ids)

    print(f"\nTraining file: {args.train_file}")
    print(f"Vocab size:    {args.vocab_size}")
    print("\nDemo example")
    print("-------------")
    print("Input text: ", example)
    print("Encoded IDs:", ids)
    print("Decoded:    ", decoded)


def display_voc(tokenizer: BPETokenizer, num_to_show: int=50) -> None:
    """Displays a random `num_to_show` sample of new tokens from a tokenizer"""
    vocab = tokenizer.vocab
    new_tokens = list(vocab.values())[256:]
    num_new_tok = len(new_tokens)

    if num_new_tok == 0:
        raise ValueError(f"No new tokens to display (vocab_size={len(vocab)}).")
    
    num_to_show = min(num_to_show, num_new_tok)

    display_tokens = sample(new_tokens, num_to_show)

    for i in range(num_to_show):
        print(display_tokens[i].decode("utf-8", errors="backslashreplace"))


def main() -> None:
    args = parse_args()
    tokenizer = training_step(args)
    demo(args, tokenizer)
    display_voc(tokenizer)    

if __name__ == "__main__":
    main()