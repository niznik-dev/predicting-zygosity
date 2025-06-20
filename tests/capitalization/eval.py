import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Paths
model_path = "/scratch/gpfs/niznik/zyg_out_finetune-five/epoch_499/"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prepare 10 words of length 6
# words = [
#     "planet", "rocket", "silver", "forest", "animal",
#     "bridge", "circle", "friend", "guitar", "hunter"
# ]

words = [
    "apple", "table", "chair", "plant", "grape",
    "house", "light", "mouse", "river", "stone"
]

for word in words:
    # Tokenize single input
    input_ids = tokenizer(word, return_tensors="pt").input_ids.to(device)
    
    # Run model
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]

    # Compute full softmax probabilities.
    softmax_probs = torch.softmax(logits, dim=0)

    # Get token id for the capitalized word
    capitalized = word.capitalize()
    capitalized_tokens = tokenizer(capitalized, add_special_tokens=False).input_ids

    print(capitalized_tokens)

    if len(capitalized_tokens) == 1:
        cand_prob = softmax_probs[capitalized_tokens[0]].item()
        print(f"{word} -> {capitalized}: {cand_prob:.4f}")
    else:
        print(f"{word} -> {capitalized}: Multi-token case")
        # Handle multi-token by looking at the first token's probability
        first_token_prob = softmax_probs[capitalized_tokens[0]].item()
        print(f"  First token probability: {first_token_prob:.4f}")

