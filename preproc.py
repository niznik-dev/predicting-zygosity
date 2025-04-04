# Option flags
add_truth = False           # When True, add the truth ("The answer is X") into the input text.
absolute_difference = False  # When True, use absolute difference; when False, use the raw twin values.

import pandas as pd
import json
import numpy as np
import sys

folder = str(sys.argv[1]) or "/home/ar0241/scratch/twins/"

# Load the CSV file.
file_path = f"{folder}twindat_sim_100k_24.csv"
df = pd.read_csv(file_path)

# Identify traits (assumes trait columns end with '.1' and '.2').
traits = sorted(set(col[:-2] for col in df.columns if col.endswith('.1')))

# If absolute_difference is True, compute the absolute difference for each trait.
if absolute_difference:
    for trait in traits:
        col1 = trait + '.1'
        col2 = trait + '.2'
        df[f"diff_{trait}"] = np.abs(df[col1] - df[col2])

# Create the instruct-format data.
def row_to_instruct(row):
    input_lines = []
    for trait in traits:
        if absolute_difference:
            # Use the computed absolute difference.
            diff_val = row[f"diff_{trait}"]
            input_lines.append(f"{trait}: {diff_val:.2f}")
        else:
            # Use raw values for each twin.
            val1 = row[trait + '.1']
            val2 = row[trait + '.2']
            input_lines.append(f"{trait}: {val1:.2f}, {val2:.2f}")
    # Join each trait information with commas.
    input_text = ", ".join(input_lines)

    # Determine output based on monozygotic status.
    output_text = "1" if row['zyg'] == 1 else "0"

    # If add_truth is True, append the truth to the input text.
    if add_truth:
        input_text += f" The answer is {output_text}"


    return {"instruction": "", "input": input_text, "output": output_text}

# Apply the transformation to create instruct-formatted data.
instruct_data = df.apply(row_to_instruct, axis=1).tolist()

# Shuffle the data (ensuring reproducibility).
np.random.seed(42)
np.random.shuffle(instruct_data)

# Split dataset: 80% train, 10% validation, 10% evaluation.
total_samples = len(instruct_data)
train_split = int(total_samples * 0.8)
val_split = int(total_samples * 0.9)

train_data = instruct_data[:train_split]
val_data = instruct_data[train_split:val_split]
eval_data = instruct_data[val_split:]

# Save the split datasets as JSON files.
with open(f"{folder}ptwindat_train.json", "w") as f:
    json.dump(train_data, f, indent=4)

with open(f"{folder}ptwindat_val.json", "w") as f:
    json.dump(val_data, f, indent=4)

with open(f"{folder}ptwindat_eval.json", "w") as f:
    json.dump(eval_data, f, indent=4)

print("Saved train (80%), validation (10%), and evaluation (10%) datasets.")

