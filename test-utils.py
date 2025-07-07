import os

import torch

import numpy as np

from utils.llm_utils import *

# ! ----------------------------- Magic Numbers -----------------------------

# Directories
RUN_NAME="100k-20epoch" # name of folder with checkpoints

BASE_DIR="/home/drigobon/scratch/"
BASE_MODEL_PATH=f"{BASE_DIR}/torchtune_models/Llama-3.2-1B-Instruct"
DATA_PATH=f"{BASE_DIR}/zyg-in/ptwindat_eval.json"
ADAPTER_PATH=None #f"{BASE_DIR}/zyg-out/{RUN_NAME}/epoch_19/" # Set to None to use base model without adapter
SAVE_PATH=f"{BASE_DIR}/out-test-llm-utils/"


# Tokenization Params.
BATCH_SIZE=4 # Batch size
USE_CHAT_TEMPLATE=True # Whether to use chat template for prompts

# Generation Params.
#   These parameters are passed to the model.generate() method in get_next_tokens().
#   See https://huggingface.co/docs/transformers/main_classes/text_generation#transformers
GENERATE_ARGS={
    "max_new_tokens": 20, # Maximum number of new tokens to generate
    "do_sample": True, # Whether to sample from the distribution
    "num_return_sequences": 2, # Number of sampled sequences to return
    "temperature": 1.0, # Temperature for sampling
}

# Data Loading Params
NUM_OBS=2 # Set to None to load all observations from the eval file


# ! ----------------------------- End Magic Numbers -----------------------------



# ----------------------------- Directory Setup -----------------------------

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Ensure the model path exists
if not os.path.exists(BASE_MODEL_PATH):
    raise FileNotFoundError(f"Base model path does not exist: {BASE_MODEL_PATH}")

# Ensure the evaluation data path exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Evaluation data path does not exist: {DATA_PATH}")

# Ensure the adapter path exists if specified
if ADAPTER_PATH is not None and not os.path.exists(ADAPTER_PATH):
    raise FileNotFoundError(f"Adapter path does not exist: {ADAPTER_PATH}")


# ----------------------------- Model Setup -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load prompts and targets
prompts, targets = load_prompts_and_targets(DATA_PATH, num_obs=NUM_OBS)
ids = [f"obs_{i}" for i in range(len(prompts))]  # Generate IDs for each observation

# Load model and tokenizer
tokenizer, model = load_model(BASE_MODEL_PATH, adapter_path=ADAPTER_PATH)
model.to(device)


# ----------------------------- Token Generation -----------------------------

# Test Token Generation
print("\n\nTesting token generation...")
generated_tokens = get_next_tokens(model, tokenizer, prompts, 
                                    batch_size=BATCH_SIZE,
                                    use_chat_template=USE_CHAT_TEMPLATE,
                                    **GENERATE_ARGS)

# Move to CPU
generated_tokens = generated_tokens.detach().cpu()  

# Print Generated Tokens
for i in range(generated_tokens.shape[0]):
    if len(generated_tokens.shape) > 2: # if num_return_sequences>1 and do_sample=True
        print(f"\n\nPrompt: \"{prompts[i]}\":")
        print("Generated Tokens:")            
        for j in range(generated_tokens.shape[1]):
            print(f"    #{j}: {tokenizer.decode(generated_tokens[i,j,:], skip_special_tokens=True)}")
    else:
        print(f"\n\nPrompt \"{prompts[i]}\": {tokenizer.decode(generated_tokens[i], skip_special_tokens=True)}")


# ----------------------------- Logits -----------------------------

# Test Logit Extraction
print("\n\nTesting logit extraction...")

logits = get_logits(model, tokenizer, prompts, 
                    batch_size=BATCH_SIZE)
print(f"Logits shape: {logits.shape}")

# Move to CPU
logits = logits.detach().cpu() 

logit_save_path = f"{SAVE_PATH}/logits.h5"
save_tensor_with_ids(logit_save_path, logits, ids)

logits_loaded, ids_loaded, _ = load_tensor_with_ids(logit_save_path, dataset_name="tensor", ids_name="ids_dim0")


# Verify loaded logits match original
assert np.array_equal(logits, logits_loaded), "Loaded logits do not match original logits"

print("Logits saved and loaded successfully")







prompt_tmp = ['What is the capital of France?']

logits = get_logits(model, tokenizer, prompt_tmp, use_chat_template=False)
probs = torch.softmax(logits, dim=-1)

vals, inds = torch.topk(probs, 10)
# Print top 10 tokens and their probabilities
print("\nTop 10 tokens and their probabilities:")
for val, ind in zip(vals[0], inds[0]):
    token = tokenizer.decode(ind.item(), skip_special_tokens=True)
    print(f"Token: {token}, Probability: {val.item():.4f}")





# ----------------------------- Embeddings -----------------------------

# Test Embedding Extraction (without pooling)
print("\n\nTesting embedding extraction...")

embed, mask = get_embeddings(model, tokenizer, prompts, pool = None,
                            batch_size=BATCH_SIZE, return_mask=True)
print(f"Embeddings shape: {embed.shape}")

# Move to CPU
embed = embed.detach().cpu() 
mask = mask.detach().cpu() if mask is not None else None

embed_save_path = f"{SAVE_PATH}/embeddings.h5"
save_tensor_with_ids(embed_save_path, embed, ids, attention_mask=mask)

embed_loaded, ids_loaded, mask_loaded = load_tensor_with_ids(embed_save_path)

# Verify loaded embeddings match original
assert np.array_equal(embed, embed_loaded), "Loaded embeddings do not match original embeddings"

# Verify loaded mask matches original
assert np.array_equal(mask, mask_loaded), "Loaded attention mask does not match original attention mask"

print("Embeddings saved and loaded successfully")


# ----------------------------- Pooling -----------------------------

pool_types = ["mean", "mean_non_padding", "median", "first", "last", "last_non_padding"]

for pool in pool_types:
    print(f"\n\nTesting pooling method: {pool}")
    
    # Pool embeddings
    pooled_embed = pool_hidden_states(embed, pool=pool, attention_mask=mask)
    print(f"Pooled embeddings shape: {pooled_embed.shape}")

    # Move to CPU
    pooled_embed = logits.detach().cpu() 

    # Save pooled embeddings
    pooled_save_path = f"{SAVE_PATH}/pooled_embeddings_{pool}.h5"
    save_tensor_with_ids(pooled_save_path, pooled_embed, ids)
    
    # Load pooled embeddings
    pooled_loaded, ids_loaded, _ = load_tensor_with_ids(pooled_save_path)

    # Verify loaded pooled embeddings match original
    assert np.array_equal(pooled_embed, pooled_loaded), f"Loaded pooled embeddings for {pool} do not match original"

    print(f"Embeddings of type {pool} saved and loaded successfully")



print('\n\nComplete!')