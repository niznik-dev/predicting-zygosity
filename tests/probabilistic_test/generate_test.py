import itertools
import json
import numpy as np
import sys
import os

# Parse inputs
outdir = "/home/ar0241/scratch/stochastic"
n = 10  # ibit length
p = 0.5
test_size = 100

# Total number of unique sequences = 2^n
total_sequences = 2 ** n
train_size = total_sequences - test_size

# Set random seed for reproducibility
np.random.seed(42)

# Generate all binary sequences of length n
sequences = [''.join(seq) for seq in itertools.product("01", repeat=n)]

# Shuffle and assign labels
np.random.shuffle(sequences)
data = [{"instruction": "", "input": seq, "output": str(int(np.random.rand() < p))} for seq in sequences]

# Split disjointly
train_data = data[:train_size]
test_data = data[train_size:]

# Save
os.makedirs(outdir, exist_ok=True)
with open(f"{outdir}/prob_train.json", "w") as f:
    json.dump(train_data, f, indent=4)
with open(f"{outdir}/prob_eval.json", "w") as f:
    json.dump(test_data, f, indent=4)

print(f"✅ Generated {train_size} training and {test_size} test examples (n={n}, p={p})")
