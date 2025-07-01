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



def load_prompts_and_targets(eval_file, num_obs):
    """
    Loads prompts and target outputs from a JSON evaluation file.
    Args:
        eval_file (str): Path to the JSON file containing evaluation data. The file should contain a list of dictionaries,
            each with at least the keys "input" (prompt) and "output" (target).
        num_obs (int or None): Number of observations to load. If None or negative, defaults to the smaller of 20 or the
            total number of examples in the file.
    Returns:
        tuple:
            - prompts (list of str): List of prompt strings extracted from the evaluation data.
            - targets (list of str): List of target output strings (stripped of leading/trailing whitespace) corresponding to the prompts.
    Example:
        prompts, targets = load_prompts_and_targets("eval_data.json", 10)
    """
    # ─── Load evaluation data ─────────────────────────────────────────────────────
    with open(eval_file, "r") as f:
        eval_data = json.load(f)

    # ─── Extract prompts & targets for desired number of observations ─────────────
    if num_obs is None or num_obs < 0:
        num_obs = min(len(eval_data))  
    prompts = [ex["input"] for ex in eval_data][0:num_obs]
    targets = [ex["output"].strip() for ex in eval_data][0:num_obs]

    return prompts, targets


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
        "--eval_file",
        type=str,
        required=True,
        help="Path to JSON file of test examples; each entry must have 'input' and 'output' ('0' or '1').",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        help="Path to the PEFT adapter folder (e.g. epoch_9/ containing adapter_config.json + adapter_model.safetensors)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="predictions.csv",
        help="Path to the CSV file that will be written. Each row will be: predicted_label,true_label",
    )
    parser.add_argument(
        "--output_embed",
        type=str,
        default='None',
        help="Path to the JSON file to output embeddings. If 'None', embeddings are not saved.",
    )
    parser.add_argument(
        "--num_obs",
        type=int,
        default=-1,
        help="Number of observations to evaluate. Defaults to the smallest of either: 20, or the number of rows in '--eval_file'.",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    # ─── Load tokenizer + model ───────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, device_map= "auto")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
    )
    if args.adapter_path is not None:
        print(f'Using fine-tuned model with adapter at:{args.adapter_path}')
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        print(f'Using base model (no fine-tuning) at: {args.base_model_path}')
        model = base_model # don't use the fine-tuned model if there's no path to the fine-tuned one

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Potentially the source of randomness...? Why is this necessary?
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))




    # ─── Load evaluation data ─────────────────────────────────────────────────────
    (prompts, targets) = load_prompts_and_targets(args.eval_file, args.num_obs)
    

    # ─── Precompute token IDs for " 1" (▁1) and " 0" (▁0) ───────────────────────────
    # LLaMA‐3.2’s tokenizer emits a space‐prefixed token for digits.
    tok_id_space1 = tokenizer(" 1", add_special_tokens=False)["input_ids"][0]
    tok_id_space0 = tokenizer(" 0", add_special_tokens=False)["input_ids"][0]

    # ─── Preallocate array for embeddings (only if saving them)  ───────────────────────────
    if args.output_embed is not None:
        # Dummy forward pass to get embedding dimension
        dummy_input = tokenizer(prompts[0], return_tensors="pt", padding=True)
        dummy_output = model(**dummy_input.to(device), output_hidden_states=True, return_dict=True)
        embedding_dim = dummy_output.hidden_states[-1].shape[-1]

        embedding_matrix = np.zeros((args.num_obs, embedding_dim), dtype=np.float32)


    # ─── Inference loop ─────────────────────────────────────────────────────────
    for i, (prompt, true_label) in enumerate(tqdm(zip(prompts, targets), total=len(prompts), desc="Evaluating")):
        inputs = tokenizer(prompt, return_tensors="pt", padding = True)
        prompt = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        input_ids = input_ids.to(device)

        with torch.no_grad():
            attention_mask = input_ids.ne(tokenizer.pad_token_id).float()
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True, 
                return_dict=True
            )

            if args.output_embed is not None:
                # To Try:
                # embeddings = model.tok_embeddings(input_ids)

                # ─── Extract token-level embeddings ───────────────────────────
                hidden_states = outputs.hidden_states[-1]  # Last layer, shape: (1, seq_len, hidden_dim)

                # ─── Mean Pooling (excluding pad tokens) Justified? Ask Keyon ───────────────────────────
                attention_mask = attention_mask.float()
                embeddings = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1)

                embedding_vector = embeddings[0].cpu().numpy()  # Shape: (hidden_dim,)

                # Add Embedding to matrix
                embedding_matrix[i,:] = embedding_vector

            # ─── Generate output text as before ───────────────────────────
            generated_text = model.generate(
                input_ids, 
                max_new_tokens=12, 
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id)
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

                # ! NOTE: NOT CURRENTLY WRITING TO THE MAIN CSV (due to large runs)
                #writer.writerow([predicted, next_token, raw_p1, raw_p0, true_label, output_text])
    # ! See above -- not currently writing
    #print(f"✅ Wrote predictions and true labels to: {args.output_csv}")


    # ─── Write embeddings matrix to json ───────────────────────────────────────────────────────
    if args.output_embed is not None:
        with open(args.output_embed, "w") as f:
            json.dump(embedding_matrix.tolist(), f)
        print(f"✅ Wrote embeddings and true labels to: {args.output_embed}")



if __name__ == "__main__":
    main()