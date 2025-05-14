import json
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score
from tqdm import tqdm

model_path = "/home/ar0241/scratch/twins/output_mar25/epoch_0"
input_prefix = "/home/ar0241/scratch/twins/"
output_prefix = "/home/ar0241/scratch/twins/"
eval_file = "/home/ar0241/scratch/stochastic/prob_eval.json"
true_p = 0.5
tolerance = 0.1

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval().cuda()

# Load eval data
with open(eval_file) as f:
    data = json.load(f)

targets, predictions = [], []

for example in tqdm(data, desc="Evaluating"):
    prompt = example["input"]
    true_label = example["output"].strip()

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1]

    probs = torch.softmax(logits, dim=0)
    p1 = probs[tokenizer("1", add_special_tokens=False)["input_ids"][0]].item()
    p0 = probs[tokenizer("0", add_special_tokens=False)["input_ids"][0]].item()

    pred = "1" if p1 >= p0 else "0"
    predictions.append(pred)
    targets.append(true_label)

# Compute accuracy
acc = accuracy_score(targets, predictions)
print(f"\n✅ Accuracy on test set: {acc:.3f}")
print(f"🎯 Expected under null (random): {true_p:.3f}")
print(f"⚠️ Tolerance threshold: ±{tolerance:.3f}")

# Check for leakage or underfitting
if abs(acc - true_p) > tolerance:
    if acc > true_p:
        print("🚨 WARNING: Accuracy above random — possible data leakage or overfitting")
    else:
        print("🚨 WARNING: Accuracy below expected — possible underfitting or config issue")
else:
    print("✅ Accuracy is within expected bounds under random labeling")
