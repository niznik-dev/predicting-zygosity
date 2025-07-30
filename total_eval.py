# This script loads either a standalone fine-tuned model or a base model with a LoRA/PEFT adapter.
# Usage examples and output file descriptions are provided in argparse help options.

import argparse
import datetime
import json
import math
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    roc_auc_score
)
from sklearn.calibration import calibration_curve

from utils import llm_utils

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Evaluate model predictions on a dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model or base model directory.")
    parser.add_argument("--input_prefix", type=str, required=True, help="Directory containing the input data file.")
    parser.add_argument("--input_filename", type=str, required=True, help="Input filename (e.g., ptwindat_eval.json).")
    parser.add_argument("--adapter_path", type=str, default=None, help="Directory of the adapter weights; will automatically set output_prefix when provided.")
    parser.add_argument("--output_prefix", type=str, default=None, help="Directory to save output results; only needed if adapter_path is None.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for loss averaging (default: 1).")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use (default: 1).")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length for tokenization (default: 2048).")

    args = parser.parse_args()

    model_path = args.model_path
    input_prefix = args.input_prefix
    input_filename = args.input_filename
    adapter_path = args.adapter_path
    output_prefix = args.output_prefix or args.adapter_path  # If adapter_path is provided, use it as output_prefix
    batch_size = args.batch_size
    num_gpus = args.num_gpus
    max_length = args.max_length

    eval_dataset_path = f"{input_prefix}/{input_filename}"

    if adapter_path is None and output_prefix is None:
        raise ValueError("Either adapter_path or output_prefix must be specified.")
    elif adapter_path and output_prefix:
        print("Both adapter_path and output_prefix are specified. Using adapter_path for output prefix.")
        output_prefix = adapter_path

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a compatible GPU and the necessary drivers installed.")

    # TODO - when OSSC is back, check how I'm setting the tokenizer path
    tokenizer, model = llm_utils.load_model(
        model_path=model_path,
        adapter_path=adapter_path,
        torch_dtype=torch.bfloat16,
    )

    prompts, targets, _ = llm_utils.load_prompts_and_targets(
        eval_file=eval_dataset_path
    )

    # Define candidate tokens for binary classification.
    candidates = ["1", "0"]
    # For single-token candidates, take the first token id.
    candidate_token_ids = {
        cand: tokenizer(cand, add_special_tokens=False)["input_ids"][0]
        for cand in candidates
    }

    predictions = []
    losses = []
    additional_details = []  # To store candidate probabilities per example

    # TODO - investigate if we should add back max_length vs abandon it
    logits = llm_utils.get_logits(model, tokenizer, prompts,
        batch_size=batch_size, use_chat_template=False,
    )

    probs = torch.softmax(logits, dim=0)

    cand1_raw_probs = probs[:, candidate_token_ids["1"]]
    cand0_raw_probs = probs[:, candidate_token_ids["0"]]
    candidate_prob_sum = cand1_raw_probs + cand0_raw_probs

    cand1_norm_probs = cand1_raw_probs / candidate_prob_sum
    cand0_norm_probs = cand0_raw_probs / candidate_prob_sum

    # Predicted probs is just the normalized probability of candidate "1"
    predicted_probs = cand1_norm_probs.tolist()
    # Predictions need to be a string because targets are strings and sklearn metrics expect string labels
    predictions = ["1" if cand1_norm_probs[i] >= cand0_norm_probs[i] else "0" for i in range(len(cand1_norm_probs))]
    target_probs = [cand1_norm_probs[i] if targets[i] == "1" else cand0_norm_probs[i] for i in range(len(targets))]
    losses = [-math.log(float(tp)) for tp in target_probs]

    for i in range(0, 5):
        detail = {
            "p(1) (norm)": float(cand1_norm_probs[i]),
            "p(0) (norm)": float(cand0_norm_probs[i]),
            "p(0) + p(1) (raw)": float(candidate_prob_sum[i]),
        }
        additional_details.append(detail)

    targets_np = np.array([1 if t == "1" else 0 for t in targets])

    # Compute classification metrics.
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, pos_label="1", average="binary")
    recall = recall_score(targets, predictions, pos_label="1", average="binary")
    f1 = f1_score(targets, predictions, pos_label="1", average="binary")
    cm = confusion_matrix(targets, predictions, labels=["1", "0"])
    report = classification_report(targets, predictions, labels=["1", "0"])

    # From here on, the metrics want targets to be integers :-|
    # Compute Brier score.
    targets_int = [1 if t == "1" else 0 for t in targets]
    brier = brier_score_loss(targets_int, predicted_probs)

    # Compute AUC score.
    auc = roc_auc_score(targets_int, predicted_probs)

    # Compute calibration curve (Murphy Curve) using sklearn.
    fraction_of_positives, mean_predicted_value = calibration_curve(targets_int, predicted_probs, n_bins=10)

    # Plot and save the calibration (Murphy) curve.
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibration curve")
    plt.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve (Murphy Curve)")
    plt.legend()
    murphy_curve_filename = f"{output_prefix}/{datetime.datetime.now().strftime('%Y-%m-%d')}_murphy_curve.png"
    plt.savefig(murphy_curve_filename)
    plt.close()

    # Save evaluation results to a text file with additional details.
    results_filename = f"{output_prefix}/{datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')}_evalresults.txt"
    with open(results_filename, "w") as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"Precision (Positive=1): {precision:.4f}\n")
        f.write(f"Recall (Positive=1): {recall:.4f}\n")
        f.write(f"F1 Score (Positive=1): {f1:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
        f.write("\nClassification Report:\n")
        f.write(report + "\n\n")
        f.write(f"Brier Score: {brier:.4f}\n")
        f.write(f"AUC Score: {auc:.4f}\n")
        f.write(f"Murphy Curve saved to: {murphy_curve_filename}\n")

        # Record details for the first 5 examples.
        f.write("Example Details (first 5 examples):\n")
        for i, detail in enumerate(additional_details):
            f.write(f"Example {i+1}:\n")
            f.write(f"  p(1) (norm): {detail['p(1) (norm)']:.4f}\n")
            f.write(f"  p(0) (norm): {detail['p(0) (norm)']:.4f}\n")
            f.write(f"  p(0) + p(1) (raw): {detail['p(0) + p(1) (raw)']:.4f}\n")
            f.write("\n")

    # TODO - we've discussed making a data frame at the end

if __name__ == "__main__":
    main()
