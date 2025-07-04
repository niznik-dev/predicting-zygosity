import os
import json
import numpy as np

import torch
from torch import nn

import h5py

from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


'''
Notes:
    - **kwargs functionality built into get_logits() and get_embeddings() methods. Need to check difference between model() and model.generate() methods...
    - Example usage in 'test-utils.py' / 'test-utils.slurm'.
'''

'''
TODO:
    - Use a dataloader?
    - Wrap load_model differently for base and PEFT models
    - Emphasizing accessibility / usability for newer users.
    - Functions should check incoming / outcoming things! Maybe not for internal functions...?
    - New function to return distribution of prompt lengths
    - Wrap pool_hidden_states() differently for ones requiring att. mask and not.
    - 
'''

def load_prompts_and_targets(eval_file: str, num_obs: int = None) -> tuple[list[str], list[str]]:
    """
    Loads prompts and target outputs from a JSON file.
    
    Args:
        eval_file (str): Path to the JSON file containing data. The file should contain a list of dictionaries,
            each with at least the keys "input" (prompt) and "output" (target).
        num_obs (int): Optional, defaults to None. Number of observations to load.
            If None defaults to the total number of examples in the file.
    
    Returns:
        tuple:
            - prompts (list of str): List of prompt strings extracted from the data.
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
    Automatically adds a padding token if the tokenizer does not have one.
    
    Args:
        base_model_path (str): The base model path
        tokenizer_path (str): Optional, defaults to None. The path to the tokenizer. If None, uses base_model_path.
        adapter_path (str): Optional, defaults to None. The path to the PEFT adapter.
    
    Returns:
        tuple:
            - tokenizer (AutoTokenizer): Loaded tokenizer.
            - model (nn.Module): Loaded model.
        
    Notes: 
        - Does not send to device automatically.
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


def tokenize_prompts(tokenizer: AutoTokenizer, prompts: list[str],
                    use_chat_template = True, max_length = None) -> dict:
    """    
    Tokenizes the given prompts using the specified tokenizer.
    
    Args:
        tokenizer (AutoTokenizer): The tokenizer to use for encoding prompts.
        prompts (list[str]): The list of prompts to tokenize.
        use_chat_template (bool): Optional, defaults to True. Whether to use chat template for encoding.
        max_length (int or None): Optional, defaults to None. Maximum length of the tokenized sequences. 
                If None, uses the max length in batch and pads to the longest prompt length.
    
    Returns:
        dict: A dictionary containing the tokenized prompts, with keys 'input_ids', 'attention_mask'.

    Notes: 
        - Does not send to device automatically.
    """

    if max_length is None:
        padding = True
    else:
        padding = 'max_length'

    if use_chat_template:
        # Apply chat template to prompts
        chat_prompts = [[{"role": "user", "content": prompt}] for prompt in prompts]

        inputs = tokenizer.apply_chat_template(chat_prompts, tokenize=True, 
                                                add_generation_prompt=True, 
                                                padding=padding, max_length=max_length,
                                                return_tensors="pt", return_dict=True, return_attention_mask=True)
    else:
        inputs = tokenizer(prompts, return_tensors="pt",
                            padding=padding, max_length=max_length)
    return inputs


def get_logits(model: nn.Module, tokenizer: AutoTokenizer, prompts: list[str],
                    use_chat_template = True, batch_size = 4,
                    **kwargs) -> torch.Tensor:
    """
    Get the logits for the given prompts and targets.

    Args:
        model (nn.Module): The model to use for inference.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding prompts.
        prompts (list[str]): The list of prompts to evaluate.
        use_chat_template (bool): Optional, defaults to True. Whether to use chat template for encoding.
        batch_size (int): Optional, defaults to 4. The batch size to use for inference.
        kwargs (dict): Additional keyword arguments to pass to model() method.

    Returns:
        torch.Tensor: The logits for the prompts. Shape: (batch_size, vocab_size). Correspond to the last non-padding token in each prompt.
    
    Notes: 
        - Outputs are NOT sent to CPU.
    """

    logits = None

    model.eval()  # Ensure the model is in evaluation mode
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i + batch_size]

            batch_inputs = tokenize_prompts(tokenizer, batch_prompts, use_chat_template=use_chat_template)
            batch_inputs = batch_inputs.to(model.device)

            outputs = model(**batch_inputs, **kwargs)
            # outputs.logits of shape: (batch_size, max_len, vocab_size)

            # Get logits of last NON-padding token (not really needed when inputs are of same length)
            input_lengths = (batch_inputs["attention_mask"].sum(dim=1) - 1)  # subtract 1 for zero-based indexing
            
            batch_logits = outputs.logits[torch.arange(len(batch_prompts)), input_lengths] # shape: (batch_size, vocab_size)

            # Concatenate
            if logits is None:
                logits = batch_logits
                # shape: (batch_size, vocab_size)
            else:
                logits = torch.cat((logits, batch_logits), dim=0)
                # shape: (batch_size*i, vocab_size)

    return logits


def get_next_tokens(model: nn.Module, tokenizer: AutoTokenizer, prompts: list[str],
                    use_chat_template = True, only_new_tokens = True, batch_size = 4,
                    **kwargs) -> torch.Tensor:
    """
    Get the next token predictions for the given prompts.

    Args:
        model (nn.Module): The model to use for inference.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding prompts.
        prompts (list[str]): The list of prompts to evaluate.
        use_chat_template (bool): Optional, defaults to True. Whether to use chat template for encoding.
        only_new_tokens (bool): Optional, defaults to True. If True, only return the newly generated tokens, excluding the input tokens.
        batch_size (int): Optional, defaults to 4. The batch size to use for inference.
        kwargs (dict): Additional keyword arguments to pass to model.generate() method.
            For details see: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers

    Returns:
        torch.Tensor: The next token predictions for the prompts. Includes the original input tokens if only_new_tokens is False.
            Shape: Either of:
                - (batch_size, seq_len) if num_return_sequences==1 (in **kwargs)
                - (batch_size, num_return_sequences, seq_len<=max_new_tokens) if num_return_sequences>1 (in **kwargs).
    
    Notes: 
        - Outputs are NOT sent to CPU.
    """

    generated_tokens = None

    model.eval()  # Ensure the model is in evaluation mode
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i + batch_size]

            batch_inputs = tokenize_prompts(tokenizer, batch_prompts, use_chat_template=use_chat_template)
            batch_inputs = batch_inputs.to(model.device)

            batch_generated_tokens = model.generate(**batch_inputs, **kwargs)

            if only_new_tokens:
                # Remove the input tokens from the generated tokens
                input_length = batch_inputs["input_ids"].shape[1]
                batch_generated_tokens = batch_generated_tokens[:, input_length:]

            # Reshape if num_return_sequences>2
            num_return_sequences = kwargs.get("num_return_sequences", 1)
            if num_return_sequences > 1:
                batch_generated_tokens = batch_generated_tokens.reshape(len(batch_prompts), num_return_sequences, -1) 
                # shape: (batch_size, num_return_sequences, seq_len<=max_new_tokens)

            # Concatenate
            if generated_tokens is None:
                generated_tokens = batch_generated_tokens
                # shape: (batch_size, vocab_size)
            else:
                generated_tokens = torch.cat((generated_tokens, batch_generated_tokens), dim=0)

    return generated_tokens


def get_embeddings(model: nn.Module, tokenizer: AutoTokenizer, prompts: list[str], 
                    use_chat_template = True, pool = "last",
                    batch_size = 4, return_mask = False, 
                    **kwargs) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Get the embeddings for the given prompts.

    Args:
        model (nn.Module): The model to use for inference.
        tokenizer (AutoTokenizer): The tokenizer to use for encoding prompts.
        prompts (list[str]): The list of prompts to evaluate.
        use_chat_template (bool): Optional, defaults to True. Whether to use chat template for encoding.
        pool (str): Optional, defaults to "last". Pooling method for the hidden states. See pool_hidden_states() for details.
        batch_size (int): Optional, defaults to 4. The batch size to use for inference.
        return_mask (bool): Optional, defaults to False. Whether to return the attention mask associated with the prompts.
        kwargs (dict): Additional keyword arguments to pass to model() method.

    Returns:
        tuple:
            - torch.Tensor: The embeddings for the prompts.
            - torch.Tensor: The attention mask for the prompts if return_mask is True, else None.
    
    Notes: 
        - Outputs are NOT sent to CPU.
        - If pool is None, expect OOM... Returns the full hidden states tensor of shape (B, L, T, H) where:
            - B: Batch size.
            - L: Number of layers.
            - T: Sequence length (number of tokens).
            - H: Hidden size.
        - If pool is not None, returns the pooled embeddings of shape (B, L, H) where:
            - B: Batch size.
            - L: Number of layers.
            - H: Hidden size.
    """

    # Extract longest prompt length in batch to pad uniformly (necessary for embedding tensors to be of same dimensions)
    # Only used if pool is None, otherwise max_length is set to None
    if pool is None:
        T = max(len(prompt) for prompt in prompts) 
    else:
        T = None

    pooled_embeds = None
    attention_mask = None

    model.eval()  # Ensure the model is in evaluation mode
    
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[i:i + batch_size]

            batch_inputs = tokenize_prompts(tokenizer, batch_prompts, use_chat_template=use_chat_template, max_length=T)
            batch_inputs = batch_inputs.to(model.device)
            batch_mask = batch_inputs["attention_mask"]

            outputs = model(**batch_inputs, output_hidden_states=True, **kwargs)

            batch_embed = torch.stack(outputs.hidden_states, dim = 1) # shape: (B, L, T, H)

            # Pool hidden states over the token dimension
            batch_pooled_embed = pool_hidden_states(batch_embed, pool=pool, attention_mask=batch_mask) 
            # shape: (B, L, H), or (B, L, T, H) if pool is None

            if pooled_embeds is None:
                pooled_embeds = batch_pooled_embed
            else:
                pooled_embeds = torch.cat((pooled_embeds, batch_pooled_embed), dim=0)
                # shape: (B*i, L, H), or (B*i, L, T, H) if pool is None

            if return_mask:
                if attention_mask is None:
                    attention_mask = batch_mask
                else:
                    attention_mask = torch.cat((attention_mask, batch_mask), dim=0)
                    # shape: (B*i, T)

    return pooled_embeds, attention_mask


def pool_hidden_states(hidden_states: torch.Tensor, 
                        pool: str = "last",
                        attention_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Pool hidden states over the token dimension.

    Args:
        hidden_states (torch.Tensor): Shape of (B, L, T, H). Third dimension !!MUST!! be token dimension.
            - B: Batch size.
            - L: Number of layers.
            - T: Sequence length (number of tokens).
            - H: Hidden size.
        pool (str): Optional, defaults to "last". Pooling method across token dimension.
            - None: No pooling, returns the hidden states as is. Not recommended -- memory will be sad :(
            - "mean": Mean pooling over the token dimension.
            - "mean_non_padding": Mean pooling over the token dimension, ignoring padding tokens.
            - "median": Median pooling over the token dimension.
            - "first": Use the first token's embedding.
            - "last": Use the last token's embedding (including padding)
            - "last_non_padding": Use the last non-padding token's embedding.
        attention_mask (torch.Tensor): Optional, defaults to None. Shape of (B, T). Mask to indicate which tokens are padding.
            Note: Required if pool is "last_non_padding" or "mean_non_padding". Ignored otherwise.

    Returns:
        torch.Tensor: Pooled embeddings of shape (B, L, H)

    Notes: 
        - Outputs are NOT sent to CPU.
    """

    if pool is None:
        # No pooling, return hidden states as is
        return hidden_states

    with torch.no_grad():
        if pool == "mean":
            pooled = hidden_states.mean(axis=2)

        elif pool == "mean_non_padding":
            assert attention_mask is not None, "attention_mask is required for pool = 'mean_non_padding' "

            expanded_mask = attention_mask[:, None, :, None] # shape: (B, 1, T, 1)

            masked_hidden_states = hidden_states * expanded_mask
            pooled = masked_hidden_states.sum(axis=2) / expanded_mask.sum(axis=2)

        elif pool == "median":
            pooled = torch.median(hidden_states, axis=2)[0] # Get only median values, not indices

        elif pool == "first":
            pooled = hidden_states[:, :, 0, :]

        elif pool == "last":
            pooled = hidden_states[:, :, -1, :]

        elif pool == "last_non_padding":
            assert attention_mask is not None, "attention_mask is required for pool = 'last_non_padding' "

            input_lengths = (attention_mask.sum(dim=1) - 1)
            pooled = hidden_states[torch.arange(hidden_states.shape[0]), : , input_lengths]

        else:
            raise ValueError(f"Unsupported pool type: {pool}")

    return pooled # shape: (B, L, H)


def save_tensor_with_ids(filename, tensor, ids, 
                        attention_mask = None, dataset_name="tensor", ids_name="ids_dim0"):
    """
    Saves a tensor and its associated dim-0 identifiers to an HDF5 file.

    Args:
        filename (str): Path to the HDF5 file to create.
        tensor (torch.Tensor): The tensor to save. 
        ids (list or np.ndarray): Identifiers associated with dimension 0.
        attention_mask (torch.Tensor): Optional, defaults to None. Tensor of attention masks corresponding to model inputs from which tensor is derived.
            If None, will not save attention masks.
        dataset_name (str): Optional, defaults to 'tensor'. Name of the dataset for the tensor.
        ids_name (str): Optional, defaults to 'ids_dim0'. Name of the dataset for the identifiers.

    Returns:
        None
    """

    assert len(tensor) == len(ids), "Length of tensor and ids must match"
    if attention_mask is not None:
        assert len(tensor) == len(attention_mask), "Length of tensor and attention_mask must match"

    tensor = np.asarray(tensor)
    ids = np.asarray(ids, dtype='S')  # store as byte strings
    if attention_mask is not None:
        attention_mask = np.asarray(attention_mask)

    if tensor.shape[0] != len(ids):
        raise ValueError("Length of ids must match size of tensor along dimension 0")

    with h5py.File(filename, 'w') as f:
        f.create_dataset(dataset_name, data=tensor)
        f.create_dataset(ids_name, data=ids)
        if attention_mask is not None:
            f.create_dataset("attention_mask", data=attention_mask)

    print(f"Saved tensor of shape {tensor.shape}{', attention_mask' if attention_mask is not None else ''} and ids to {filename}")


def load_tensor_with_ids(filename, dataset_name="tensor", ids_name="ids_dim0") -> tuple[np.ndarray, list[str], np.ndarray | None]:
    """
    Loads a tensor and its associated dim-0 identifiers from an HDF5 file.

    Args:
        filename (str): Path to the HDF5 file.
        dataset_name (str): Optional, defaults to 'tensor'. Name of the tensor dataset.
        ids_name (str): Optional, defaults to 'ids_dim0'. Name of the identifiers dataset.

    Returns:
        tuple: 
            - tensor (np.ndarray): Object of name 'dataset_name' from the HDF5 file.
            - ids (list of str): Object of name 'ids_name' from the HDF5 file.
            - attention_mask (np.ndarray): Object of name 'attention_mask' from HDF5 file.
                Only returned if 'attention_mask' dataset exists in the file, else returns None.
    
    Note: Nothing is sent to any device!
    """

    with h5py.File(filename, 'r') as f:
        tensor = f[dataset_name][:]
        ids = f[ids_name][:]
        ids = [id.decode('utf-8') if isinstance(id, bytes) else str(id) for id in ids]
        if "attention_mask" in f:
            attention_mask = f["attention_mask"][:]
        else:
            attention_mask = None

    print(f"Loaded tensor of shape {tensor.shape}{', attention_mask' if attention_mask is not None else ''} and ids from {filename}")

    return tensor, ids, attention_mask

