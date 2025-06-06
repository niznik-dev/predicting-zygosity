import os
import json
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def compute_parity(bitstr: str) -> str:
    return str(bitstr.count("1") % 2)

def classify_probs(p0: float, p1: float) -> str:
    return "1" if p1 >= p0 else "0"

def run_smoke_tests():
    # parity
    assert compute_parity("")     == "0"
    assert compute_parity("1011") == "1"
    assert compute_parity("1001") == "0"
    # classifier
    assert classify_probs(0.3,0.7) == "1"
    assert classify_probs(0.6,0.4) == "0"
    print("✅ Self-checks passed.\n")

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate fine-tuned parity model")
    p.add_argument("--model_path",   type=str, required=True,
                   help="Path to your fine-tuned model directory")
    p.add_argument("--eval_file",    type=str, required=True,
                   help="JSON file of parity test examples")
    p.add_argument("--noise_p",      type=float, required=True,
                   help="Label-noise probability used at generation time")
    p.add_argument("--results_csv",  type=str, default="results.csv",
                   help="CSV to append noise_p,accuracy pairs to")
    p.add_argument("--probabilistic", action="store_true",
                   help="Instead of accuracy, report mean P(true_label)")
    p.add_argument("--prob_threshold", type=float, default=0.5,
                   help="Threshold for mean-prob check")
    return p.parse_args()

def main():
    args = parse_args()
    run_smoke_tests()

    # load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.eval().cuda()

    # load data
    with open(args.eval_file) as f:
        data = json.load(f)

    preds, targets, true_probs = [], [], []
    for ex in tqdm(data, desc="Evaluating"):
        seq = ex["input"].strip()
        true_lbl = ex["output"].strip()
        targets.append(true_lbl)

        # forward
        input_ids = tokenizer(seq, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            logits = model(input_ids).logits[0, -1]
            probs  = torch.softmax(logits, dim=-1)

        # extract P(0), P(1)
        id0 = tokenizer("0", add_special_tokens=False)["input_ids"][0]
        id1 = tokenizer("1", add_special_tokens=False)["input_ids"][0]
        p0, p1 = probs[id0].item(), probs[id1].item()

        if args.probabilistic:
            true_probs.append(p1 if true_lbl=="1" else p0)

        else:
            preds.append(classify_probs(p0,p1))

    # compute & report
    if args.probabilistic:
        mean_p = float(np.mean(true_probs))
        print(f"\n🎯 Mean P(true_label): {mean_p:.3f}")
        print("✅ Above threshold" if mean_p>=args.prob_threshold
              else "⚠️ Below threshold")
        metric = mean_p
    else:
        acc = accuracy_score(targets, preds)
        print(f"\n✅ Accuracy: {acc:.3f}")
        # your existing checks
        if acc==1.0:
            print("💾 Perfect recall (memorization)")
        elif abs(acc-0.5)<0.02:
            print("🌐 Generalization ≈ random guess")
        else:
            print("⚠️ Unexpected (check for leakage or bugs)")
        metric = acc

    # ——— log to CSV ———
    os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)
    with open(args.results_csv, "a") as out:
        out.write(f"{args.noise_p:.6f},{metric:.6f}\n")
    print(f"➡️ Logged to {args.results_csv}: noise_p={args.noise_p}, metric={metric:.6f}")

if __name__ == "__main__":
    main()
