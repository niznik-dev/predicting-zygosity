import os

import torch

import numpy as np

from tqdm import tqdm

from utils.llm_utils import load_model, load_prompts_and_targets, get_next_tokens, get_logits, get_embeddings, pool_hidden_states, save_tensor_with_ids, load_tensor_with_ids



# ! ----------------------------- Magic Numbers -----------------------------

# Directories
RUN_NAME="100k-20epoch" # name of folder with checkpoints

BASE_DIR="/home/drigobon/scratch/"
BASE_MODEL_PATH=f"{BASE_DIR}/torchtune_models/Llama-3.2-1B-Instruct"
DATA_PATH=f"{BASE_DIR}/zyg-in/ptwindat_eval.json"
MAX_EPOCH=20
ADAPTER_PATHS=[None]+[f"{BASE_DIR}/zyg-out/{RUN_NAME}/epoch_{i}/" for i in range(MAX_EPOCH)] # Set to None to use base model without adapter
SAVE_PATHS=[f"{BASE_DIR}/zyg-out/{RUN_NAME}/hidden_states/base_model/"]+[f"{BASE_DIR}/zyg-out/{RUN_NAME}/hidden_states/epoch_{i}/" for i in range(MAX_EPOCH)] # Directories to save hidden states


# Tokenization Params.
BATCH_SIZE=4 # Batch size
USE_CHAT_TEMPLATE=True # Whether to use chat template for prompts


# Data Loading Params
NUM_OBS=200 # Set to None to load all observations from the eval file


# Embedding Pooling Params
POOL_TYPES = ['mean_non_padding', 'last_non_padding']


# ! ----------------------------- End Magic Numbers -----------------------------


# Load prompts and targets
prompts, targets = load_prompts_and_targets(DATA_PATH, num_obs=NUM_OBS)
ids = [f"obs_{i}" for i in range(len(prompts))]  # Generate IDs for each observation

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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

    # Get embeddings
    embed, mask = get_embeddings(model, tokenizer, prompts, 
                            batch_size=BATCH_SIZE, return_mask=True)

    # Save
    embed_save_path = f"{SAVE_PATH}/raw_embeddings.h5"
    save_tensor_with_ids(embed_save_path, embed, ids, attention_mask=mask)

    # Pooling Options
    for POOL_TYPE in POOL_TYPES:
        pooled_embed = pool_hidden_states(embed, pool=POOL_TYPE, attention_mask=mask)

        # Save pooled embeddings
        pooled_save_path = f"{SAVE_PATH}/pooled_{POOL_TYPE}.h5"
        save_tensor_with_ids(pooled_save_path, pooled_embed, ids)



