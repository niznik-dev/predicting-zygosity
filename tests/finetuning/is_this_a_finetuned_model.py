#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import torch.nn.functional as F

def compute_perplexity(logits, input_ids):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction='mean'
    )
    return torch.exp(loss).item()

def compare_models(base_model_path, adapter_path, prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    base_model.eval()

    # Load adapter model
    adapter_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    adapter_model = PeftModel.from_pretrained(adapter_model, adapter_path)
    adapter_model.eval()

    with torch.no_grad():
        base_logits = base_model(input_ids).logits
        adapter_logits = adapter_model(input_ids).logits

    base_ppl = compute_perplexity(base_logits, input_ids)
    adapter_ppl = compute_perplexity(adapter_logits, input_ids)
    mse_diff = F.mse_loss(base_logits, adapter_logits).item()

    print("\n=== Model Comparison ===")
    print(f"Input: {prompt}")
    print(f"Base Model Perplexity:    {base_ppl:.3f}")
    print(f"Adapter Model Perplexity: {adapter_ppl:.3f}")
    print(f"Logit MSE Difference:     {mse_diff:.6f}")

    if abs(base_ppl - adapter_ppl) < 0.01 and mse_diff < 1e-4:
        print("⚠️  Adapter model appears very similar to base model — fine-tuning might not have taken effect.")
    else:
        print("✅ Adapter model is different — likely fine-tuned.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare base vs adapter model using perplexity and logits.")
    parser.add_argument("--base_model_path", required=True, help="Path to base model")
    parser.add_argument("--adapter_path", required=True, help="Path to adapter (PEFT) model")
    parser.add_argument("--prompt", required=True, help="Input string to evaluate")
    args = parser.parse_args()

    compare_models(args.base_model_path, args.adapter_path, args.prompt)
