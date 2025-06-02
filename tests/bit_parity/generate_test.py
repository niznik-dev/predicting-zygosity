import itertools
import json
import numpy as np
import os
import argparse

def compute_parity(bitstr: str) -> str:
    """0 if even number of 1s, 1 if odd."""
    return str(bitstr.count("1") % 2)

def split_data(sequences, test_size, memorize, seed):
    """
    If memorize=True, test is sampled from train (memory test).
    Else, train/test are disjoint (generalization test).
    """
    rng = np.random.RandomState(seed)
    rng.shuffle(sequences)
    total = len(sequences)

    if memorize:
        train = sequences[: total - test_size]
        # sample from train without replacement
        test = list(rng.choice(train, size=test_size, replace=False))
    else:
        train = sequences[: total - test_size]
        test  = sequences[total - test_size:]
    return train, test

def write_dataset(seqs, out_path, rng, noise_p):
    """
    Label each seq by parity, flip with prob=noise_p, and dump JSON.
    """
    data = []
    for seq in seqs:
        true_lbl = compute_parity(seq)
        # flip with probability noise_p
        if rng.rand() < noise_p:
            label = "1" if true_lbl == "0" else "0"
        else:
            label = true_lbl
        data.append({
            "instruction": "",
            "input": seq,
            "output": label
        })
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"🖋️   Wrote {len(data)} examples (noise_p={noise_p}) to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate parity train/test datasets (with optional label noise)."
    )
    parser.add_argument("--n",         type=int,   default=10,
                        help="Length of each bit sequence")
    parser.add_argument("--test_size", type=int,   default=100,
                        help="Number of examples in the test set")
    parser.add_argument("--memorize",  action="store_true",
                        help="If set, test examples are drawn from train (memory test).")
    parser.add_argument("--noise_p",   type=float, default=0.0,
                        help="Probability to flip the true parity label (0–1)")
    parser.add_argument("--seed",      type=int,   default=42,
                        help="Random seed for split & noise")
    parser.add_argument("--outdir",    type=str,   default="/home/ar0241/scratch/bit_parity/",
                        help="Directory to write parity_train.json & parity_test.json")

    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)

    # Prepare output dir
    os.makedirs(args.outdir, exist_ok=True)

    # Generate all 2^n bitstrings
    all_seqs = ["".join(bits) for bits in itertools.product("01", repeat=args.n)]

    # Split into train vs test
    train_seqs, test_seqs = split_data(all_seqs, args.test_size, args.memorize, args.seed)

    # Write JSON datasets (with noise)
    write_dataset(train_seqs,
                  os.path.join(args.outdir, "parity_train.json"),
                  rng, args.noise_p)
    write_dataset(test_seqs,
                  os.path.join(args.outdir, "parity_test.json"),
                  rng, args.noise_p)

    print("✅ Dataset generation complete.")
