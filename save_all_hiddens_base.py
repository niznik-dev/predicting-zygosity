import os

import torch

import numpy as np

from tqdm.auto import tqdm

from utils.llm_utils import *


# ! ----------------------------- Magic Numbers -----------------------------

# ----------- Common Directories -----------
RUN_NAME="1B-100k-20epoch" # name of folder with checkpoints
BASE_DIR="/home/drigobon/scratch/"
BASE_MODEL_PATH=f"{BASE_DIR}/torchtune_models/Llama-3.2-1B-Instruct"
DATA_PATH=f"{BASE_DIR}/BoL-in/book_of_life_paraphrases.1K.json"
CHECKPOINT_BASE_DIR=f"{BASE_DIR}/BoL-out/{RUN_NAME}/" # Base directory for checkpoints


# ----------- Models and Paths to Save -----------
# If using only base model:
ADAPTER_PATHS = [None]
SAVE_PATHS = [f"{BASE_DIR}/preprompt_test/hidden_states/"]


# ----------- Tokenization Params. -----------
BATCH_SIZE = 8 # Batch size
USE_CHAT_TEMPLATE = True # Whether to use chat template for prompts


# ----------- Data Loading Params -----------
NUM_OBS = None # Set to None to load all observations from the eval file


# ----------- Embedding & Pooling Params -----------
RETURN_MASK = False # Whether to return the attention mask (should pretty much always be false, unless we want to have the length of each prompt.)
POOL_TYPES = ['mean_non_padding', 'last_non_padding']

# ----------- Embedding & Pooling Params -----------
PREPROMPTS={
    'default': '',
    # 'DR': "You are a highly educated medical specialist. The following is a short description of a person, who is currently living in the Netherlands. Your task is to determine if this individual exhibits any risk of cardiovascular disease. Provide a response of 1 if they do, and 0 if they do not. Respond with only a single number, either 1 or 0. Do not provide any additional information or explanation.",
    # 'KS': "You are an AI assistant that is a social science specialist. Summarize the main information about this person for registry-level retrieval task. Be concise, focus on factual information.",
    # 'MS': "You are a very smart language model. I'd like you to summarize the information in this story about a person. Please focus on what is important about the essence of the person.  Now what is one word you'd use to describe the essence of this person.",
    # 'AM': "Pretend you've been fine tuned for a task. Try to figure out what is the right target grouping of these people.",
    # 'MN': "Hey buddy, do your best!",
}

# ! ----------------------------- End Magic Numbers -----------------------------

'''
Notes:
    - Runs forward passes multiple times to pool in different ways... Takes len(POOL_TYPES) times longer than it should... But, saves on memory! Not an issue if only one type of pooling is used.
'''


print("------------ Starting: Extract Hidden States ------------")

# Load prompts and targets
prompts, targets = load_prompts_and_targets(DATA_PATH, num_obs=NUM_OBS)
ids = [f"obs_{i}" for i in range(len(prompts))]  # Generate IDs for each observation

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Loop over candidate model checkpoints (each corresponding to a different adapter)
# We do this first to avoid loading the model multiple times, which is expensive.
for i in range(len(ADAPTER_PATHS)):

    ADAPTER_PATH = ADAPTER_PATHS[i]
    print(f"\n\nUsing model with adapter {ADAPTER_PATH}")

    # Load model and tokenizer
    tokenizer, model = load_model(BASE_MODEL_PATH, adapter_path=ADAPTER_PATH)
    model.to(device)

    # Loop over preprompts
    for preprompt_name, preprompt in PREPROMPTS.items():
        print(f"\nUsing preprompt: {preprompt_name} - '{preprompt}'")

        SAVE_PATH = f"{SAVE_PATHS[i]}/{preprompt_name}/"

        # Directories
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        print(f"Saving to {SAVE_PATH}")


        # Loop over pooling types specified in POOL_TYPES
        for POOL_TYPE in POOL_TYPES:
            print(f"\nPooling type: {POOL_TYPE}")

            # Get embeddings
            embeds, mask = get_embeddings(model, tokenizer, prompts, 
                                        preprompt=preprompt,
                                        use_chat_template=USE_CHAT_TEMPLATE, pool=POOL_TYPE,
                                        batch_size=BATCH_SIZE, return_mask=RETURN_MASK)

            # Move to CPU
            embeds = embeds.detach().cpu()
            if mask is not None:
                mask = mask.detach().cpu()

            # Save embeddings
            pooled_save_path = f"{SAVE_PATH}/embeds_pooled_{POOL_TYPE}.h5"
            save_tensor_with_ids(pooled_save_path, embeds, ids)
            print(f"Saved pooled embeddings of shape {embeds.shape} to {pooled_save_path}")


print("------------ Extracting Hidden States Complete! ------------")



