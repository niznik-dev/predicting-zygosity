#!/usr/bin/env python3
import argparse
import json
import os
import torch
import numpy as np
import csv
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel  # remove this import if you are not using a PEFT adapter

def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a binary‐sequence eval set and write a CSV of (predicted, true_label)."
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the base Llama-3.2-1B-Instruct checkpoint folder",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the PEFT adapter folder (e.g. epoch_9/ containing adapter_config.json + adapter_model.safetensors)",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="Path to JSON file of test examples; each entry must have 'input' and 'output' ('0' or '1').",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="predictions.csv",
        help="Path to the CSV file that will be written. Each row will be: predicted_label,true_label",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    # ─── Load tokenizer + model ───────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, device_map= "auto")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # ─── Load evaluation data ─────────────────────────────────────────────────────
    with open(args.eval_file, "r") as f:
        eval_data = json.load(f)

    prompts = [ex["input"] for ex in eval_data][0:15]
    targets = [ex["output"].strip() for ex in eval_data][0:15]

    # ─── Precompute token IDs for " 1" (▁1) and " 0" (▁0) ───────────────────────────
    # LLaMA‐3.2’s tokenizer emits a space‐prefixed token for digits.

    tok_id_space1 = tokenizer(" 1", add_special_tokens=False)["input_ids"][0]
    tok_id_space0 = tokenizer(" 0", add_special_tokens=False)["input_ids"][0]

    # ─── Open CSV for writing ───────────────────────────────────────────────────────
    with open(args.output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["predicted_label","next_token","p1", "p0", "true_label", "output_text"])

        # ─── Inference loop ─────────────────────────────────────────────────────────
        for prompt, true_label in tqdm(zip(prompts, targets), total=len(prompts), desc="Evaluating"):
            inputs = tokenizer(prompt, return_tensors="pt")
            prompt = [{"role": "user", "content": prompt}]
            #input_ids = inputs.input_ids.to(device)
            input_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            input_ids = input_ids.to(device)

            with torch.no_grad():
                outputs = model(input_ids)
                generated_text = model.generate(input_ids, max_new_tokens=12)
                output_text = tokenizer.decode(generated_text[0])
            
            logits = outputs.logits[0, -1, :]  # next‐token logits

            probs = torch.softmax(logits, dim=0)
            next_token = tokenizer.decode(probs.argmax())
            raw_p1 = probs[tok_id_space1].item()
            raw_p0 = probs[tok_id_space0].item()
            denom = raw_p1 + raw_p0

            if denom > 0:
                norm_p1 = raw_p1 / denom
                norm_p0 = raw_p0 / denom
            else:
                norm_p1 = 0.5
                norm_p0 = 0.5

            
            predicted = "1" if norm_p1 >= norm_p0 else "0"
            writer.writerow([predicted, next_token, raw_p1, raw_p0, true_label, output_text])
            
    print(f"✅ Wrote predictions and true labels to: {args.output_csv}")


if __name__ == "__main__":
    main()

