#!/usr/bin/env python3
"""
Generate binary-sequence datasets with parity-based or probabilistic labelling.

Examples
--------
Parity dataset, 1 000 000 samples of length 5, noiseless, 1 000-example test set
    python generate.py --bit_length 5 --N 1000000 --p 0 --bit_parity True --test_size 1000

Probabilistic dataset, p = 0.5, 500 000 samples of length 8, 1 000-example test set
    python generate.py --bit_length 8 --N 500000 --p 0.5 --bit_parity False --test_size 1000
"""
import argparse
import itertools
import json
from pathlib import Path
from typing import List, Dict

import numpy as np


# -----------------------------------------------------------------------------#
# Helpers
# -----------------------------------------------------------------------------#
def compute_parity(bitstr: str) -> str:
    """Return '0' if the bit-string has an even number of “1”s, else '1'."""
    return str(bitstr.count("1") % 2)


def parse_bool(value: str) -> bool:
    """Parse booleans passed on the command line."""
    val = value.lower()
    if val in {"true", "t", "1"}:
        return True
    if val in {"false", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate binary-sequence datasets (parity or Bernoulli labelling)."
    )
    p.add_argument("--bit_length", type=int, required=True, help="Sequence length n ≥ 1")
    p.add_argument(
        "--N", type=int, required=True, help="Number of samples to generate (N ≥ 1)"
    )
    p.add_argument(
        "--test_size",
        type=int,
        required=True,
        help="Exact number of examples to put in the test set",
    )
    p.add_argument(
        "--bit_parity",
        type=parse_bool,
        required=True,
        choices=[True, False],
        help="True → parity labels (with optional noise p); False → Bernoulli(p) labels",
    )
    p.add_argument(
        "--p",
        type=float,
        default=0.0,
        help=(
            "If --bit_parity True: probability of *flipping* the true parity label.\n"
            "If False: probability of label 1 in Bernoulli labelling."
        ),
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path.cwd(),
        help="Directory for train.json and test.json",
    )
    return p.parse_args()


def label_sequences(
    sequences: List[str], bit_parity: bool, p: float, rng: np.random.RandomState
) -> List[Dict[str, str]]:
    """Attach labels to every sequence and return instruction/input/output dicts."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Probability p must be in [0, 1], got {p}")

    labelled = []
    for seq in sequences:
        if bit_parity:
            true_lbl = int(compute_parity(seq))
            lbl = true_lbl ^ int(rng.rand() < p)  # flip with prob p
        else:
            lbl = int(rng.rand() < p)
        labelled.append({"instruction": "", "input": seq, "output": str(lbl)})
    return labelled


# -----------------------------------------------------------------------------#
# Script entry point
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    args = parse_args()

    n = args.bit_length
    N = args.N
    test_size = args.test_size

    assert n > 0, "--bit_length must be ≥ 1"
    assert 0 < test_size < N, "--test_size must be in (0, N)"

    rng = np.random.RandomState(args.seed)

    # ---------------------------------------------------------------------#
    # 1. Generate the population and draw N samples WITH replacement
    # ---------------------------------------------------------------------#
    universe = ["".join(bits) for bits in itertools.product("01", repeat=n)]
    sampled_seqs = rng.choice(universe, size=N, replace=True)

    # ---------------------------------------------------------------------#
    # 2. Pick EXACTLY `test_size` sample indices for the test split
    #    (no replacement → indices are unique)
    # ---------------------------------------------------------------------#
    test_indices = rng.choice(N, size=test_size, replace=False)
    test_mask = np.zeros(N, dtype=bool)
    test_mask[test_indices] = True

    # ---------------------------------------------------------------------#
    # 3. Ensure the TRAIN split contains *no* sequence that appears in TEST
    #    • Build a set of the sequences found in the test subset.
    #    • Keep only those training rows whose sequence is NOT in that set.
    # ---------------------------------------------------------------------#
    test_seqs = sampled_seqs[test_mask].tolist()
    test_seq_set = set(test_seqs)

    train_seqs = [seq for seq in sampled_seqs[~test_mask] if seq not in test_seq_set]

    # ---------------------------------------------------------------------#
    # 4. strict separation at the sequence level
    # ---------------------------------------------------------------------#
    assert test_seq_set.isdisjoint(train_seqs), "Leakage: sequence appears in both splits"

    # ---------------------------------------------------------------------#
    # 5. Label, dump, done
    # ---------------------------------------------------------------------#
    train_data = label_sequences(train_seqs, args.bit_parity, args.p, rng)
    test_data = label_sequences(test_seqs, args.bit_parity, args.p, rng)

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "train.json").write_text(json.dumps(train_data, indent=2))
    (args.outdir / "test.json").write_text(json.dumps(test_data, indent=2))

    print(
        f"✅  Generated {len(train_data)} train and {len(test_data)} test examples "
        f"(bit_parity={args.bit_parity}, p={args.p})"
    )


