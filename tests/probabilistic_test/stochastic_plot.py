import argparse
import math
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot test accuracy vs. p, with Bayes-optimal and 95% CI baselines."
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of bits (so total = 2**n possible sequences).",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        required=True,
        help="If eval_on_test=false, this is the test-set size; if eval_on_test=true, this is the *training* size, and we evaluate on the remainder.",
    )
    parser.add_argument(
        "--eval_on_test",
        type=lambda x: x.lower() == "true",
        required=True,
        help='"true" or "false". If true, then test_size refers to the *training* set and N = 2**n - test_size; otherwise N = test_size.',
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of epochs used in training (to display on the plot).",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default="/home/ar0241/scratch/twins/results.csv",
        help="Path to the CSV file containing columns [p, accuracy].",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="/home/ar0241/scratch/twins/accuracy_vs_p.png",
        help="Base path (including .png) for saving the resulting plot. A random suffix will be added automatically.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) Compute total number of sequences
    total_sequences = 2 ** args.n

    # 2) Compute effective test-set size N
    if args.eval_on_test:
        N = total_sequences - args.test_size
        if N <= 0:
            raise ValueError(
                f"With n={args.n}, total sequences = {total_sequences}, "
                f"but test_size (used as training) = {args.test_size} leaves no test samples."
            )
    else:
        N = args.test_size
        if N <= 0 or N > total_sequences:
            raise ValueError(
                f"test_size = {args.test_size} is invalid for total_sequences = {total_sequences}."
            )

    # 3) Load your CSV; assume it has exactly two columns: p and accuracy (no header)
    df = pd.read_csv(args.results_csv, names=["p", "accuracy"])
    df = df.sort_values("p").reset_index(drop=True)

    # 4) Prepare arrays for Bayes-optimal and its 95% CI
    z = 1.96  # for approximate 95% two-sided interval

    best_possible = []
    ci_lower = []
    ci_upper = []

    for p in df["p"]:
        # Bayes-optimal accuracy on an infinite sample = max(p, 1-p)
        if p <= 0.5:
            bayes_acc = 1.0 - p
        else:
            bayes_acc = p

        # Standard error of sample proportion p_hat for N draws
        se = math.sqrt(p * (1 - p) / N)

        # 95% CI around bayes_acc:
        low = bayes_acc - z * se
        high = bayes_acc + z * se

        # Clip into [0,1]
        low = max(0.0, low)
        high = min(1.0, high)

        best_possible.append(bayes_acc)
        ci_lower.append(low)
        ci_upper.append(high)

    df["bayes_optimal"] = best_possible
    df["ci_lower"] = ci_lower
    df["ci_upper"] = ci_upper

    # 5) Construct a unique output filename by appending a random integer suffix
    base, ext = os.path.splitext(args.output_png)
    rand_suffix = random.randint(0, 999999)
    unique_output = f"{base}_{rand_suffix}{ext}"

    # 6) Plotting
    plt.figure(figsize=(8, 5))
    # a) Empirical accuracy vs p
    plt.plot(
        df["p"],
        df["accuracy"],
        marker="o",
        label="Empirical accuracy",
        linewidth=1.5,
        alpha=0.8,
    )

    # b) Random baseline at 0.5
    plt.hlines(
        0.5,
        df["p"].min(),
        df["p"].max(),
        linestyles="dashed",
        color="gray",
        label="Random baseline (0.5)",
    )

    # c) Bayes-optimal curve
    plt.plot(
        df["p"],
        df["bayes_optimal"],
        linestyle="--",
        color="tab:green",
        label="Bayes-optimal (max[p, 1-p])",
    )

    # d) Confidence-interval bounds
    plt.plot(
        df["p"],
        df["ci_lower"],
        linestyle=":",
        color="tab:green",
        label="CI lower (95%)",
    )
    plt.plot(
        df["p"],
        df["ci_upper"],
        linestyle=":",
        color="tab:green",
        label="CI upper (95%)",
    )

    # e) Add annotation (n, N, epochs) in bottom‐right corner
    ax = plt.gca()
    textstr = f"n = {args.n}\nN = {N}\nepochs = {args.epochs}"
    ax.text(
        0.95,
        0.05,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5),
    )

    plt.xlabel("Label-generation probability $p$")
    plt.ylabel("Test-set accuracy")
    plt.title(
        f"Accuracy vs. $p$ for random-label test (n={args.n}, N_test={N})"
    )
    plt.legend(loc="best", fontsize="small")
    plt.tight_layout()
    plt.savefig(unique_output, dpi=150)
    print(f"Saved plot to: {unique_output}")
    plt.show()


if __name__ == "__main__":
    main()
