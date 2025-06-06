#!/usr/bin/env python3
import argparse
import itertools
import json
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(
        description="Generate random binary sequences with labels"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/home/ar0241/scratch/stochastic",
        help="where to write prob_train.json & prob_eval.json",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="length of each binary sequence",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.5,
        help="probability of a positive (1) label",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=100, 
        help="how many examples in the test set",
    )
    args = parser.parse_args()

    # now these are defined
    n = args.n
    p = args.p 
    test_size = args.test_size 
    outdir = args.outdir
    
    # Total number of unique sequences = 2^n
    total_sequences = 2 ** n 
    train_size = total_sequences - test_size

    # reproducible shuffling
    np.random.seed(42)

    # generate all sequences
    sequences = [''.join(seq) for seq in itertools.product("01", repeat=n)]
    np.random.shuffle(sequences)

    # assign labels
    data = [
        {"instruction": "",
         "input": seq,
         "output": str(int(np.random.rand() < p))}
        for seq in sequences
    ]

    train_data = data[:train_size]
    test_data  = data[train_size:]

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "prob_train.json"), "w") as f:
        json.dump(train_data, f, indent=4)
    with open(os.path.join(outdir, "prob_eval.json"), "w") as f:
        json.dump(test_data, f, indent=4)

    print(f"✅ Generated {train_size} train / {test_size} test "
          f"(n={n}, p={p}, outdir={outdir})")

if __name__ == "__main__":
    main()
