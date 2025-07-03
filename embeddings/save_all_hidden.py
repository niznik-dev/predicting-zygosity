from transformers import LlamaTokenizer, LlamaModel
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

save_file_name = "outputs/book_of_life_hidden_states_all_layers_sample_3_longer.gemma.pt"
df = pd.read_csv("outputs/book_of_life_sample_3_longer.csv")
df.head()


model_id = "google/gemma-3-27b-it" # "mistralai/Mistral-Small-24B-Base-2501" #"google/gemma-3-27b-it"
local_dir = "./gemma-3-27b-it" #"./Mistral-Small-24B-Base-2501" #"./gemma-3-27b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=local_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    output_hidden_states=True,
    cache_dir=local_dir,
    torch_dtype="auto", 
    device_map="auto",    
    local_files_only=True 
)
model.eval()

texts = df["text"].tolist()

print('Device:')
print(next(model.parameters()).device)

batch_size = 1
all_hidden_states_dict = {}

with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]
        person_ids = df.iloc[i:i + batch_size]["person_id"].tolist()

        # inputs = tokenizer(batch_texts, return_tensors="pt", 
        #                    padding="max_length",
        #                    truncation=True, 
        #                    max_length=45).to(model.device)
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True).to(model.device)

        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # list of (1, T, H) tensors

        stacked = torch.stack(hidden_states)[:, 0]  # shape: (L, T, H)

        person_id = i
        all_hidden_states_dict[person_id] = stacked.cpu().half()

# Save full dictionary
with open(save_file_name, "wb") as f:
    pickle.dump(all_hidden_states_dict, f)

print(f"Saved {len(all_hidden_states_dict)} entries with shape: {list(all_hidden_states_dict.values())[0].shape}")
