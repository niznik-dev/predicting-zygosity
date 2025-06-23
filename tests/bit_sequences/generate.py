#!/usr/bin/env python3
"""
Generate binary sequence datasets with either parity-based or probabilistic labeling to any outdir.

Usage examples:
  # Parity dataset (no noise), 2^5=32 unique sequences 
  python generate.py --bit_length 5 --N 1000000 --p 0 --bit_parity True --test_size 1000

  # Probabilistic dataset, p=0.3, 2^8=256 unique sequences
  python generate.py --bit_length 8 --N 500000 --p 0.5 --bit_parity False --test_size 1000
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np

def compute_parity(bitstr: str) -> str:
    """Return '0' if even number of '1's, else '1'."""
    return str(bitstr.count('1') % 2)

def parse_bool(value: str) -> bool:
    """Parse a boolean CLI argument (True/False)."""
    val = value.lower()
    if val in ('true', 't', '1'):
        return True
    if val in ('false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected (True or False), got '{value}'")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate binary sequences labeled by parity or Bernoulli(p)."
    )
    parser.add_argument(
        "--bit_length", type=int, required=True,
        help="Length of each bit sequence (n ≥ 1)."
    )
    parser.add_argument(
        "--N", type=int, required=True,
        help="Total number of examples to generate (N ≥ 2^bit_length)."
    )
    parser.add_argument(
        "--test_size", type=int, required=True,
        help="Number of examples in the test set."
    )
    parser.add_argument(
        "--bit_parity", type=parse_bool, required=True, choices=[True, False],
        help=(
            "Choose labeling mode: True for parity-based (with noise p), "
            "False for Bernoulli(p) labeling."
        )
    )
    parser.add_argument(
        "--p", type=float, default=0.0,
        help=(
            "Probability parameter: if --bit_parity True, noise probability to flip parity labels; "
            "if False, probability of a positive (1) label."
        )
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path.cwd(),
        help="Directory to write train.json and test.json."
    )
    return parser.parse_args()

def label_sequences(sequences, bit_parity: bool, p: float, rng: np.random.RandomState):
    """
    Label each sequence by parity (with noise) or by Bernoulli(p).
    Returns a list of dicts with keys: instruction, input, output.
    """
    data = []
    for seq in sequences:
        if bit_parity:
            true_lbl = int(compute_parity(seq))
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"Noise probability p must be in [0,1], got {p}")
            flip = rng.rand() < p
            lbl = true_lbl ^ int(flip)
            label = str(lbl)
        else:
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"Probability p must be in [0,1], got {p}")
            label = str(int(rng.rand() < p))
        data.append({"instruction": "", "input": seq, "output": label})
    return data

if __name__ == "__main__":
    args = parse_args()
    n = args.bit_length
    N = args.N
    test_size = args.test_size
    p = args.p
    bit_parity = args.bit_parity
    seed = args.seed
    outdir = args.outdir

    assert n > 0, f"bit_length must be ≥ 1, got {n}"
    assert 0 < test_size < N, f"test_size must be in (0, N), got {test_size} with N={N}"

    rng = np.random.RandomState(seed)

    # Generate all unique sequences
    all_seqs = ["".join(bits) for bits in itertools.product("01", repeat=n)]

    # Sample N sequences with replacement
    sampled_seqs = rng.choice(all_seqs, size=N, replace=True)

    # Identify unique sequences from the sample
    unique_sampled = list(set(sampled_seqs))
    rng.shuffle(unique_sampled)

    # Choose unique test sequences
    num_test_unique = min(test_size, len(unique_sampled))
    test_unique = set(unique_sampled[:num_test_unique])

    # Allocate each sample to test/train set by its sequence
    train_seqs = []
    test_seqs = []
    test_count = 0

    for seq in sampled_seqs:
        if seq in test_unique and test_count < test_size:
            test_seqs.append(seq)
            test_count += 1
        else:
            train_seqs.append(seq)

    # Label both sets
    train_data = label_sequences(train_seqs, bit_parity, p, rng)
    test_data = label_sequences(test_seqs, bit_parity, p, rng)

    # Write output
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    with open(outdir / "test.json", "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"✅ Generated {len(train_data)} train and {len(test_data)} test examples (bit_parity={bit_parity}, p={p})")

