
import sys
import os
import time
import json
import numpy as np

import torch
from omegaconf import DictConfig, ListConfig

from torch import nn
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel  # remove this import if you are not using a PEFT adapter

from tqdm import tqdm


def load_prompts_and_targets(eval_file: str, num_obs: int = None) -> tuple[list[str], list[str]]:
    """
    Loads prompts and target outputs from a JSON evaluation file.
    Args:
        eval_file (str): Path to the JSON file containing evaluation data. The file should contain a list of dictionaries,
            each with at least the keys "input" (prompt) and "output" (target).
        num_obs (int or None): Number of observations to load. If None defaults to the total number of examples in the file.
    Returns:
        tuple:
            - prompts (list of str): List of prompt strings extracted from the evaluation data.
            - targets (list of str): List of target output strings (stripped of leading/trailing whitespace) corresponding to the prompts.
    """
    
    # ─── Load evaluation data ─────────────────────────────────────────────────────
    with open(eval_file, "r") as f:
        eval_data = json.load(f)

    # ─── Extract prompts & targets for desired number of observations ─────────────
    if num_obs is None:
        num_obs = len(eval_data)
    else:
        assert num_obs > 0, "num_obs must be greater than 0"    

    eval_data = eval_data[:num_obs]

    prompts = [ex["input"] for ex in eval_data]
    targets = [ex["output"].strip() for ex in eval_data]

    return prompts, targets

def load_model(model_path: str, tokenizer_path: str = None, adapter_path: str = None) -> tuple[AutoTokenizer,nn.Module]:
    """
    Loads a model checkpoint with folder specified by model_path. If adapter_path is None, returns base model.
    Args:
        base_model_path (str): The base model path
        tokenizer_path (str): Optional. Defaults to None (and uses same path as base_model). The path to the tokenizer.
        adapter_path (str): Optional. Defaults to None. The path to the PEFT adapter.
    Returns:
        tuple: A tuple containing the tokenizer and loaded model.
    """

    # Load tokenizer
    if tokenizer_path is None:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load models
    assert os.path.exists(model_path), f"Model path {model_path} does not exist."
    base_model = AutoModelForCausalLM.from_pretrained(model_path)

    if adapter_path is not None:
        assert os.path.exists(adapter_path), f"Adapter path {adapter_path} does not exist."
        model_out = PeftModel.from_pretrained(base_model, adapter_path)        
    else:
        model_out = base_model

    # Add padding token if necessary
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model_out.resize_token_embeddings(len(tokenizer))

    return tokenizer, model_out

def get_logits(model: nn.Module, tokenizer: AutoTokenizer, prompts: list[str],
                    use_chat_template = True,
                    **kwargs) -> torch.Tensor:
    """
    Get the logits for the given prompts and targets.
    Args:
        model (nn.Module): The model to use for inference.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding prompts.
        prompts (list[str]): The list of prompts to evaluate.
        use_chat_template (bool): Whether to use chat template for encoding.
        kwargs: Additional keyword arguments to pass to model() method.
    Returns:
        torch.Tensor: The logits for the prompts. Shape: (batch_size, vocab_size). Correspond to the last non-padding token in each prompt.
    """
    if use_chat_template:
        # Apply chat template to prompts
        chat_prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]
        inputs = tokenizer.apply_chat_template(chat_prompts, tokenize=True, add_generation_prompt=True, padding=True,
                                                return_tensors="pt", return_dict=True, return_attention_mask=True)
        inputs = inputs.to(device)
    else:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = inputs.to(device) # potentially will fail? if device not defined globally...

    b = inputs["input_ids"].shape[0]  # batch size

    with torch.no_grad():
        model.eval()  # Ensure the model is in evaluation mode
        outputs = model(**inputs, **kwargs)
        # outputs.logits of shape: (b, seq_len, vocab_size)

        # Get logits of last NON-padding token (not really needed when inputs are of same length)
        input_lengths = (inputs["attention_mask"].sum(dim=1) - 1)  # subtract 1 for zero-based indexing
        
        logits = outputs.logits[torch.arange(b), input_lengths]
        # shape: (b, vocab_size)

    return logits

def get_next_tokens(model: nn.Module, tokenizer: AutoTokenizer, prompts: list[str],
                    use_chat_template = True, only_new_tokens = True,
                    **kwargs) -> torch.Tensor:
    """
    Get the next token predictions for the given prompts.
    Args:
        model (nn.Module): The model to use for inference.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding prompts.
        prompts (list[str]): The list of prompts to evaluate.
        use_chat_template (bool): Whether to use chat template for encoding.
        only_new_tokens (bool): If True, only return the newly generated tokens, excluding the input tokens.
        kwargs: Additional keyword arguments to pass to model.generate() method.
    Returns:
        torch.Tensor: The next token predictions for the prompts. Includes the original input tokens if only_new_tokens is False.
    """
    if use_chat_template:
        # Apply chat template to prompts
        chat_prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]
        inputs = tokenizer.apply_chat_template(chat_prompts, tokenize=True, add_generation_prompt=True, padding=True,
                                                return_tensors="pt", return_dict=True, return_attention_mask=True)
        inputs = inputs.to(device)
    else:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = inputs.to(device) # potentially will fail? if device not defined globally...

    with torch.no_grad():
        model.eval()  # Ensure the model is in evaluation mode
        generated_tokens = model.generate(**inputs, **kwargs)
        # shape: (batch_size, seq_len<=max_new_tokens)

    if only_new_tokens:
        # Remove the input tokens from the generated tokens
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = generated_tokens[:, input_length:]

    # Reshape if num_return_sequences>2
    num_return_sequences = kwargs.get("num_return_sequences", 1)
    if num_return_sequences > 1:
        generated_tokens = generated_tokens.reshape(len(prompts), num_return_sequences, -1) 
        # shape: (batch_size, num_return_sequences, seq_len<=max_new_tokens)

    return generated_tokens

def get_embeddings(model: nn.Module, tokenizer: AutoTokenizer, prompts: list[str], 
                    use_chat_template = True, pool = False, **kwargs) -> torch.Tensor:
    """
    Get the embeddings for the given prompts.
    Args:
        model (nn.Module): The model to use for inference.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding prompts.
        prompts (list[str]): The list of prompts to evaluate.
        use_chat_template (bool): Whether to use chat template for encoding.
        pool (bool): If True, average all non-masked token embeddings. Else, returns the embedding for the last non-masked token only.
        **kwargs: Additional keyword arguments to pass to model() method.
    Returns:
        torch.Tensor: The embeddings for the prompts.
    """

    if use_chat_template:
        # Apply chat template to prompts
        chat_prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]
        inputs = tokenizer.apply_chat_template(chat_prompts, tokenize=True, add_generation_prompt=True, padding=True,
                                                return_tensors="pt", return_dict=True, return_attention_mask=True)
        inputs = inputs.to(device)
    else:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = inputs.to(device) # potentially will fail? if device not defined globally...

    d_embed = model.config.hidden_size  # hidden size of the model
    
    with torch.no_grad():
        model.eval()  # Ensure the model is in evaluation mode
        outputs = model(**inputs, output_hidden_states=True, **kwargs)
        
    last_state = outputs.hidden_states[-1]  # Get the last hidden state

    if pool:
        embeddings = (last_state * inputs['attention_mask'].unsqueeze(-1)).sum(1) / inputs['attention_mask'].unsqueeze(-1).sum(1)
    else:
        # Get the last non-masked token embedding
        input_lengths = (inputs["attention_mask"].sum(dim=1) - 1)
        embeddings = last_state[torch.arange(last_state.shape[0]), input_lengths]

    return embeddings



if __name__ == "__main__":

    # Directories
    RUN_NAME="100k-20epoch" # name of folder with checkpoints

    BASE_DIR="/home/drigobon/scratch/"
    PYTHON_FILE_PATH=f"{BASE_DIR}/embeddings-analysis/get_embeddings.py"
    BASE_MODEL_PATH=f"{BASE_DIR}/torchtune_models/Llama-3.2-1B-Instruct"
    EVAL_DATA_PATH=f"{BASE_DIR}/zyg-in/ptwindat_eval.json"
    ADAPTER_PATH=f"{BASE_DIR}/zyg-out/{RUN_NAME}/epoch_0/" # Set to None to use base model without adapter
    EMBED_SAVE_PATH=f"{BASE_DIR}/zyg-out/{RUN_NAME}/embeddings/"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load prompts and targets
    prompts, targets = load_prompts_and_targets(EVAL_DATA_PATH, num_obs=1)

    # Load model and tokenizer
    tokenizer, model = load_model(BASE_MODEL_PATH, adapter_path=ADAPTER_PATH)
    model.to(device)

    # Get Logits & Probabilities
    logits = get_logits(model, tokenizer, prompts)
    probs = torch.softmax(logits, dim=1)

    # Get generated tokens
    generated_tokens = get_next_tokens(model, tokenizer, prompts,
                                    only_new_tokens=True, use_chat_template=True,
                                    max_new_tokens=15, do_sample=True, num_return_sequences=100)

    '''
    for i in range(generated_tokens.shape[0]):
        if len(generated_tokens.shape) > 2: # if num_return_sequences>1 and do_sample=True
            print(f"Generated tokens for prompt: {prompts[i]}:\n\n")
            for j in range(generated_tokens.shape[1]):
                print(f"    #{j}: {tokenizer.decode(generated_tokens[i,j,:], skip_special_tokens=True)}")
        else:
            print(f"Generated tokens for prompt {i}: {tokenizer.decode(generated_tokens[i], skip_special_tokens=True)}")
    '''


    k = 5  # Top-k predictions

    vocab_size = len(tokenizer)
    top_k_probs, top_k_indices = torch.topk(probs, k, dim=1)

    assert generated_tokens.shape > 2, "generated_tokens should have at least 3 dimensions (batch_size, num_return_sequences, seq_len)"

    for i in range(top_k_indices.shape[0]):
        print(f"\nPrompt: {prompts[i]}")
        
        if len(generated_tokens.shape) > 2:  # if num_return_sequences>1 and do_sample=True
            freq_next_token = np.bincount(generated_tokens[i,:,0].cpu().numpy(), minlength=vocab_size)/generated_tokens.shape[1]

            for j in range(k):
                token = tokenizer.decode(top_k_indices[i, j], skip_special_tokens=True)
                prob = top_k_probs[i, j].item()
                print(f"    #{j}: {token} \n        (probability: {prob:.4f}) \n        (frequency: {freq_next_token[top_k_indices[i, j]].item():.4f})")

            
'''
    # Compare empirical distribution of generated tokens with next token probs

    for i in range(len(prompts)):
        freq_next_token = np.bincount(generated_tokens[i,:,0].cpu().numpy(), minlength=vocab_size)/generated_tokens.shape[1]
        print(f"Empirical distribution of next tokens for prompt {i}:")


    # Test on other prompts
    prompt_tmp = ['What is 1+1?',
                  'What is the capital of France?',
                  'Who wrote "To Kill a Mockingbird"?',
                  'What is the largest mammal?',
                  'What is the boiling point of water?',
                  'What is the speed of light?',
                  'What is the meaning of life?',
                  'What is the capital of Japan?',
                  'Who painted the Mona Lisa?',
                  'What is the square root of 16?',
                  'What is the chemical symbol for gold?',
                  'What is the currency of the United States?',
                  'What is the largest planet in our solar system?']
    
    generated_tokens = get_next_tokens(model, tokenizer, prompt_tmp,
                                    only_new_tokens=True, use_chat_template=True,
                                    max_new_tokens=15, do_sample=True, num_return_sequences=10)
    
    for i in range(generated_tokens.shape[0]):
        if len(generated_tokens.shape) > 2:
            print(f"Generated tokens for prompt: {prompt_tmp[i]}:\n\n")
            for j in range(generated_tokens.shape[1]):
                print(f"    #{j}: {tokenizer.decode(generated_tokens[i,j,:], skip_special_tokens=True)}")
        else:
            print(f"Generated tokens for prompt {i}: {tokenizer.decode(generated_tokens[i], skip_special_tokens=True)}")
'''