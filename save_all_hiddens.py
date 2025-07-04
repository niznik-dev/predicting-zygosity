import os

import torch

import numpy as np

from tqdm import tqdm

from utils.llm_utils import load_model, load_prompts_and_targets, get_next_tokens, get_logits, get_embeddings, pool_hidden_states, save_tensor_with_ids, load_tensor_with_ids



# ! ----------------------------- Magic Numbers -----------------------------

# Common Directories
RUN_NAME="100k-20epoch" # name of folder with checkpoints
BASE_DIR="/home/drigobon/scratch/"
BASE_MODEL_PATH=f"{BASE_DIR}/torchtune_models/Llama-3.2-1B-Instruct"
DATA_PATH=f"{BASE_DIR}/zyg-in/ptwindat_eval.json"

# Models and Paths to Save
MAX_EPOCH=20
ADAPTER_PATHS=[None]+\
    [f"{BASE_DIR}/zyg-out/{RUN_NAME}/epoch_{i}/" for i in range(MAX_EPOCH)] # List of paths to adapter checkpoints, None for base model
SAVE_PATHS=[f"{BASE_DIR}/zyg-out/{RUN_NAME}/hidden_states/base_model/"]+\
    [f"{BASE_DIR}/zyg-out/{RUN_NAME}/hidden_states/epoch_{i}/" for i in range(MAX_EPOCH)] # Directories to save hidden states


# Tokenization Params.
BATCH_SIZE=4 # Batch size
USE_CHAT_TEMPLATE=True # Whether to use chat template for prompts


# Data Loading Params
NUM_OBS=25 # Set to None to load all observations from the eval file


# Embedding Pooling Params
POOL_TYPES = ['mean_non_padding', 'last_non_padding']


# ! ----------------------------- End Magic Numbers -----------------------------

print("------------ Starting: Extract Hidden States ------------")

# Load prompts and targets
prompts, targets = load_prompts_and_targets(DATA_PATH, num_obs=NUM_OBS)
ids = [f"obs_{i}" for i in range(len(prompts))]  # Generate IDs for each observation

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Loop over candidate model checkpoints (each corresponding to a different adapter)
for i in range(len(ADAPTER_PATHS)):

    SAVE_PATH = SAVE_PATHS[i]
    ADAPTER_PATH = ADAPTER_PATHS[i]

    print(f"\n\nUsing model with adapter {ADAPTER_PATH}")
    print(f"Saving to {SAVE_PATH}")

    # Directories
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Load model and tokenizer
    tokenizer, model = load_model(BASE_MODEL_PATH, adapter_path=ADAPTER_PATH)
    model.to(device)

    # Loop over pooling types specified in POOL_TYPES
    for POOL_TYPE in POOL_TYPES:
        print(f"\nPooling type: {POOL_TYPE}")

        # Get embeddings
        embeds, mask = get_embeddings(model, tokenizer, prompts, 
                                    use_chat_template=USE_CHAT_TEMPLATE, pool=POOL_TYPE,
                                    batch_size=BATCH_SIZE, return_mask=False)

        # Move to CPU
        embeds = embeds.detach().cpu()
        if mask is not None:
            mask = mask.detach().cpu()

        # Save embeddings
        pooled_save_path = f"{SAVE_PATH}/embeds_pooled_{POOL_TYPE}.h5"
        save_tensor_with_ids(pooled_save_path, embeds, ids)
        print(f"Saved pooled embeddings of shape {embeds.shape} to {pooled_save_path}")


print("------------ Extracting Hidden States Complete! ------------")


