
import os

import numpy as np

import torch

from tqdm import tqdm

# Hack to import from parent directory
prev_wd = os.getcwd()
os.chdir("/home/drigobon/scratch/predicting-zygosity/") # Wherever llm_utils.py is located
from llm_utils import load_prompts_and_targets, load_model, get_logits, get_next_tokens, get_embeddings
os.chdir(prev_wd) # Go back to original working directory


if __name__ == "__main__":

    # ! ----------------------------- Magic Numbers -----------------------------

    # Directories
    RUN_NAME="100k-20epoch" # name of folder with checkpoints

    BASE_DIR="/home/drigobon/scratch/"
    PYTHON_FILE_PATH=f"{BASE_DIR}/embeddings-analysis/get_embeddings.py"
    BASE_MODEL_PATH=f"{BASE_DIR}/torchtune_models/Llama-3.2-1B-Instruct"
    EVAL_DATA_PATH=f"{BASE_DIR}/zyg-in/ptwindat_eval.json"
    ADAPTER_PATH=f"{BASE_DIR}/zyg-out/{RUN_NAME}/epoch_19/" # Set to None to use base model without adapter
    EMBED_SAVE_PATH=f"{BASE_DIR}/zyg-out/{RUN_NAME}/embeddings/"
    

    # Data Loading Params
    num_obs = 20 # Set to None to load all observations from the eval file

    # Generating Tokens Params
    generate_args = {
        "max_new_tokens": 15,
        "do_sample": True,
        "num_return_sequences": 5,  # Only permitted to be >1 if do_sample=True. 
        "renormalize_logits": True,  # Normalize logits to probabilities
        "temperature": 1,  # Temperature for sampling
    }
    num_runs = 100
    #   Number of token generation runs. Avoids memory issues if num_return_sequences is large. 
    #   Yields (after loop) generated_tokens of shape (num_prompts, num_return_sequences*num_runs, max_new_tokens)
    #   Only used for testing empirical probabilities of next tokens. 
    #   BUT!! num_return_sequences is still used for testing on other prompts.

    # Logits & Probabilites Params
    k = 2  # top k tokens to consider for comparing empirical probabilities with theoretical probabilities

    # Querying Test Set Params
    test_emprirical_probs = False

    # Querying Other Prompts Params
    test_on_other_prompts = True
    prompts_new = [
        'What is 1+1?',
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
        'What is the largest planet in our solar system?'
    ]

    # ! ----------------------------- End Magic Numbers -----------------------------
    

    # ------------------------------------------ Setup ------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load prompts and targets
    prompts, targets = load_prompts_and_targets(EVAL_DATA_PATH, num_obs=num_obs)

    # Load model and tokenizer
    tokenizer, model = load_model(BASE_MODEL_PATH, adapter_path=ADAPTER_PATH)
    model.to(device)




    # ------------------------------------------ Empirical vs Theoretical next-token probs ------------------------------------------

    if test_emprirical_probs:

        # Get Logits & Probabilities
        logits = get_logits(model, tokenizer, prompts)
        probs = torch.softmax(logits, dim=1)

        # Set seed
        torch.manual_seed(42)

        # Get generated tokens
        generated_tokens = None
        for i in tqdm(range(num_runs), desc="Generating tokens"):
            # Generate tokens for the prompts
            generated_tokens_tmp = get_next_tokens(model, tokenizer, prompts,
                                                    only_new_tokens=True, use_chat_template=True,
                                                    **generate_args)
            
            if generated_tokens is None:
                generated_tokens = generated_tokens_tmp
            else:
                generated_tokens = torch.cat((generated_tokens, generated_tokens_tmp), dim=1)
        


        # Compare empirical distribution of generated tokens with next token probs

        vocab_size = len(tokenizer)
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=1)

        # Check for multiple generated tokens per prompt (else no sense in comparing empirical distributions...)
        assert len(generated_tokens.shape) > 2, "generated_tokens should have at least 3 dimensions (batch_size, num_return_sequences, seq_len)"


        for i in range(top_k_indices.shape[0]):
            print(f"\nPrompt: {prompts[i]}")
            print(f"Target: {targets[i]}")
            
            if len(generated_tokens.shape) > 2:  # if num_return_sequences>1 and do_sample=True
                freq_next_token = np.bincount(generated_tokens[i,:,0].cpu().numpy(), minlength=vocab_size)/generated_tokens.shape[1]
                # 3rd dim index for generated_tokens is 0 to look ONLY at first generated token.

                for j in range(k):
                    token = tokenizer.decode(top_k_indices[i, j], skip_special_tokens=True)
                    prob = top_k_probs[i, j].item()
                    print(f"  Top #{j} Token: {token} \n    (probability: {prob:.4f}) \n    (frequency (n={generated_tokens.shape[1]}): {freq_next_token[top_k_indices[i, j]].item():.4f}, S.E.: {np.sqrt(freq_next_token[top_k_indices[i, j]] * (1 - freq_next_token[top_k_indices[i, j]]) / generated_tokens.shape[1]):.4f})")




    # ------------------------------------------ Querying Model on Other Prompts ------------------------------------------

    if test_on_other_prompts:
        # Set seed
        torch.manual_seed(42)
        generated_tokens_new = get_next_tokens(model, tokenizer, prompts_new,
                                                    only_new_tokens=True, use_chat_template=True,
                                                    **generate_args)
        
        for i in range(generated_tokens_new.shape[0]):
            if len(generated_tokens_new.shape) > 2:
                print(f"Generated tokens for prompt: {prompts_new[i]}:\n\n")
                for j in range(generated_tokens_new.shape[1]):
                    print(f"    #{j}: {tokenizer.decode(generated_tokens_new[i,j,:], skip_special_tokens=True)}")
            else:
                print(f"Generated tokens for prompt {i}: {tokenizer.decode(generated_tokens_new[i], skip_special_tokens=True)}")
        



    '''
    # For printing out generated tokens
    # Note: Commented out to avoid printing too much output...
    for i in range(generated_tokens.shape[0]):
        if len(generated_tokens.shape) > 2: # if num_return_sequences>1 and do_sample=True
            print(f"Generated tokens for prompt: {prompts[i]}:\n\n")
            for j in range(generated_tokens.shape[1]):
                print(f"    #{j}: {tokenizer.decode(generated_tokens[i,j,:], skip_special_tokens=True)}")
        else:
            print(f"Generated tokens for prompt {i}: {tokenizer.decode(generated_tokens[i], skip_special_tokens=True)}")
    '''
