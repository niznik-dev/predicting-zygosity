#!/usr/bin/env python3
import argparse
import os
import random
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot parity accuracy vs. label noise, adding best‐possible V‐curve "
                    "and stamping sample size / epochs / test size."
    )
    p.add_argument(
        "--results_csv",
        type=str,
        default="/home/ar0241/scratch/bit_parity/results.csv",
        help="Path to CSV with columns [noise_p, accuracy]."
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="/home/ar0241/scratch/bit_parity",
        help="Directory to write the timestamped plot PNG."
    )
    p.add_argument(
        "--n",
        type=int,
        required=True,
        help="Length of bit‐strings (so that sample size = 2^n)."
    )
    p.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of training epochs completed."
    )
    p.add_argument(
        "--test_size",
        type=int,
        required=True,
        help="Number of examples in the held‐out test set."
    )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 1) Read the accumulated CSV
    df = pd.read_csv(args.results_csv, names=["noise_p", "accuracy"])
    df = df.sort_values("noise_p")

    # 2) Prepare the “best‐possible” accuracy: max(1-p, p)
    noise_vals = df["noise_p"].values
    best_possible = np.maximum(1.0 - noise_vals, noise_vals)

    # 3) Plot everything
    plt.figure(figsize=(6, 4))

    # 3a) Observed accuracies
    plt.plot(
        df["noise_p"],
        df["accuracy"],
        marker="o",
        linestyle="-",
        label="Observed accuracy",
        color="tab:blue",
    )

    # 3b) Random‐guess baseline at 0.5
    plt.hlines(
        0.5,
        xmin=noise_vals.min(),
        xmax=noise_vals.max(),
        linestyles="dashed",
        label="Random baseline (0.5)",
        color="tab:gray",
    )

    # 3c) Best‐possible V‐shaped curve (no leakage)
    plt.plot(
        noise_vals,
        best_possible,
        linestyle="solid",
        marker="x",
        label="Best possible (no leakage)",
        color="tab:orange",
    )

    # 4) Labels, legend, grid, etc.
    plt.xlabel("Label‐noise probability $p$")
    plt.ylabel("Test accuracy")
    plt.title("Parity accuracy vs. label noise")
      plt.legend(loc="upper right", fontsize="small", ncol=1)
    plt.grid(alpha=0.3)

    # 5) Stamp sample size, epochs, test size in upper-left corner
    sample_size = 2 ** args.n
    stamp_text = (
        f"Sample size: 2^{args.n} = {sample_size}\n"
        f"Epochs: {args.epochs}\n"
        f"Test size: {args.test_size}"
    )
    # Place it at axes coords (0.02, 0.95), anchored top‐left
    plt.gca().text(
        0.02, 0.95, stamp_text,
        transform=plt.gca().transAxes,
        fontsize="small",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    plt.tight_layout()

    # 6) Save to a timestamped + random‐suffix PNG
    os.makedirs(args.outdir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    rand_suffix = f"{random.randint(0, 10**10 - 1):010d}"
    fname = f"accuracy_{date_str}_{rand_suffix}.png"
    outpath = os.path.join(args.outdir, fname)

    plt.savefig(outpath, dpi=150)
    print(f"✅ Plot saved to: {outpath}")
    plt.close()
