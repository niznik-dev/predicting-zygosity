from transformers import LlamaTokenizer, LlamaModel
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv("outputs/book_of_life_sample_1.csv")
df.head()


model_id = "google/medgemma-27b-text-it"
local_dir = "./medgemma-27b"
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
batch_size = 4

all_hidden_states = []  # Will become list of [num_layers x (B, T, H)]

print('Device:')
print(next(model.parameters()).device)


with torch.no_grad():
    for i in tqdm(range(0, len(texts), batch_size)):
        print(i)
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", 
                           padding="max_length",
                           truncation=True, 
                           max_length=45).to(model.device)

        outputs = model(**inputs)
        hidden_states = outputs.hidden_states  # tuple of (L+1) tensors, each (B, T, H)
        
        print(f"Batch {i}: {[h.shape for h in outputs.hidden_states]}")


        # Stack: shape becomes (L, B, T, H)
        stacked = torch.stack(hidden_states)
        all_hidden_states.append(stacked.cpu())

# Concatenate across batches: (L, total_B, T, H)
all_hidden_states_tensor = torch.cat(all_hidden_states, dim=1)
print(f"Final hidden states tensor shape: {all_hidden_states_tensor.shape}")

# stacked: list of (L, B, T, H) tensors, already collected in batches
all_hidden_states = torch.cat(all_hidden_states, dim=1)  # shape: (L, N, T, H)

# Move to CPU and convert to float16 to save space
all_hidden_states = all_hidden_states.cpu().half()

# Save to compressed npz
np.savez_compressed("outputs/book_of_life_hidden_states_sample1.npz", hidden_states=all_hidden_states.numpy())