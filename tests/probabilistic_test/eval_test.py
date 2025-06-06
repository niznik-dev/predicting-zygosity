#!/usr/bin/env python3
import argparse
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Import HuggingFace + PEFT
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned LM (PEFT adapter) on a binary sequence task"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the base Llama-3.2-1B-Instruct model (full HF checkpoint)",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the PEFT adapter folder (e.g. epoch_19/ containing adapter_config.json + adapter_model.safetensors)",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="JSON file of test examples",
    )
    parser.add_argument(
        "--true_p",
        type=float,
        required=True,
        help="expected random accuracy",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.1,
        help="allowed deviation from random accuracy",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default=None,
        help="(optional) full path to CSV where results should be appended. If unset, uses $RESULTS_CSV or 'results.csv' in cwd.",
    )
    args = parser.parse_args()

    # ─── Load base model + tokenizer from disk ────────────────────────────────────
    # NOTE: this directory must contain config.json, model.safetensors (or pytorch_model.bin pieces),
    #       tokenizer.json, tokenizer_config.json, special_tokens_map.json, etc.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        low_cpu_mem_usage=True  # optional: reduce peak CPU RAM if desired
    )

    # ─── Load the PEFT adapter on top of base ─────────────────────────────────────
    # This will merge adapter weights in memory but leave base weights untouched on disk.
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # ─── Load evaluation data ─────────────────────────────────────────────────────
    with open(args.eval_file, "r") as f:
        data = json.load(f)

    targets = []
    predictions = []
    for example in tqdm(data, desc="Evaluating"):
        prompt = example["input"]
        true_label = example["output"].strip()

        # Tokenize prompt, push to GPU if available
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.cuda() if torch.cuda.is_available() else inputs.input_ids

        # Forward pass, grab logits for last token
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0, -1]

        # Compute probability of "1" vs "0"
        probs = torch.softmax(logits, dim=0)
        token_id_1 = tokenizer("1", add_special_tokens=False)["input_ids"][0]
        token_id_0 = tokenizer("0", add_special_tokens=False)["input_ids"][0]
        p1 = probs[token_id_1].item()
        p0 = probs[token_id_0].item()

        pred = "1" if p1 >= p0 else "0"
        predictions.append(pred)
        targets.append(true_label)

    # ─── Compute accuracy ─────────────────────────────────────────────────────────
    acc = accuracy_score(targets, predictions)
    print(f"\n✅ Accuracy on test set: {acc:.3f}")

    # ─── Write out to CSV ─────────────────────────────────────────────────────────
    # Priority: --results_csv > $RESULTS_CSV env var > ./results.csv
    if args.results_csv:
        results_file = args.results_csv
    else:
        results_file = os.environ.get("RESULTS_CSV", "results.csv")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "a") as out:

            out.write(f"{args.true_p:.6f},{acc:.6f}\n")
    print(f"➡️ Logged to {results_file}: p={args.true_p:.6f}, acc={acc:.6f}")

    # ─── Print expected vs. observed, check for leakage/underfitting ─────────────
    print(f"🎯 Expected under null (random): {args.true_p:.3f}")
    print(f"⚠️ Tolerance threshold: ±{args.tolerance:.3f}")

    diff = abs(acc - args.true_p)
    if diff > args.tolerance:
        if acc > args.true_p:
            print("🚨 WARNING: Accuracy above random — possible data leakage or overfitting")
        else:
            print("🚨 WARNING: Accuracy below expected — possible underfitting or config issue")
    else:
        print("✅ Accuracy is within expected bounds under random labeling")


if __name__ == "__main__":
    main()
