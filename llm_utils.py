import os
import json

import torch

from torch import nn

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel  # remove this import if you are not using a PEFT adapter


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
        kwargs (dict): Additional keyword arguments to pass to model() method.
    Returns:
        torch.Tensor: The logits for the prompts. Shape: (batch_size, vocab_size). Correspond to the last non-padding token in each prompt.
    """
    if use_chat_template:
        # Apply chat template to prompts
        chat_prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]
        inputs = tokenizer.apply_chat_template(chat_prompts, tokenize=True, add_generation_prompt=True, padding=True,
                                                return_tensors="pt", return_dict=True, return_attention_mask=True)
        inputs = inputs.to(model.device)
    else:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device) 

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
        kwargs (dict): Additional keyword arguments to pass to model.generate() method.
    Returns:
        torch.Tensor: The next token predictions for the prompts. Includes the original input tokens if only_new_tokens is False.
            Shape: (batch_size, seq_len) if num_return_sequences==1 (in **kwargs), or (batch_size, num_return_sequences, seq_len<=max_new_tokens) if num_return_sequences>1 (in **kwargs).
    """
    if use_chat_template:
        # Apply chat template to prompts
        chat_prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]
        inputs = tokenizer.apply_chat_template(chat_prompts, tokenize=True, add_generation_prompt=True, padding=True,
                                                return_tensors="pt", return_dict=True, return_attention_mask=True)
        inputs = inputs.to(model.device)
    else:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device) 

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
        kwargs (dict): Additional keyword arguments to pass to model() method.
    Returns:
        torch.Tensor: The embeddings for the prompts.
    """

    if use_chat_template:
        # Apply chat template to prompts
        chat_prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]
        inputs = tokenizer.apply_chat_template(chat_prompts, tokenize=True, add_generation_prompt=True, padding=True,
                                                return_tensors="pt", return_dict=True, return_attention_mask=True)
        inputs = inputs.to(model.device)
    else:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device)

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


