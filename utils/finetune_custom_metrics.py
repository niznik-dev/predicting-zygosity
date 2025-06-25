"""
Custom metrics for model evaluation during fine-tuning
"""

import torch
import torch.nn.functional as F
from typing import Dict, List


def get_token_id(token: str, tokenizer=None) -> int:
    """
    Get token ID from tokenizer, raising error if not found.
    
    Args:
        token: String token to look up (e.g., "0", "1")
        tokenizer: Tokenizer to use for lookup
        
    Returns:
        Token ID
        
    Raises:
        ValueError: If tokenizer is None or token lookup fails
    """
    if tokenizer is None:
        raise ValueError(f"Tokenizer is required to look up token '{token}'")
    
    try:
        token_ids = tokenizer.encode(token)
        print(f'Token IDs for "{token}": {token_ids}')
        if not token_ids:
            raise ValueError(f"Token '{token}' not found in tokenizer vocabulary")
        # TODO - this is cludgy but I can't figure out how to disable BOS/EOS tokens!
        return token_ids[1]
    except Exception as e:
        raise ValueError(f"Failed to encode token '{token}': {e}")


def calculate_prob_0_1_sum(
    metrics: Dict[str, float],
    logits: List[torch.Tensor],
    labels: torch.Tensor,
    tokenizer=None,
    ignore_index=-100
):
    """
    Calculate average sum of p(0) + p(1) to check if model has learned 
    that 0 and 1 are the valid outputs.
    
    Args:
        logits: Raw model outputs [batch_size * seq_len, vocab_size]
        labels: Ground truth labels [batch_size * seq_len]
        tokenizer: Tokenizer to look up token IDs for "0" and "1"
        ignore_index: Index to ignore in calculations
        
    Returns:
        Average sum of p(0) + p(1) across valid tokens
    """

    # Concatenate along sequence dimension and take logits from where we think the answer is generated
    full_logits = torch.cat(logits, dim=1)

    # Target positions are the last non-ignore index in each sequence
    # Based on investigation, our prediciton will come 3 later
    # (0: BOS, 1: "Assistant", 2: EOS, 3: "/n" -> "0" or "1")
    target_positions = (labels != ignore_index).float().argmax(dim=1) + 3
    print(target_positions)
    for i in range(labels[0].size(0)):
        if labels[0, i] != ignore_index:
            print(i, labels[0, i].item(), tokenizer.decode([labels[0, i].item()]))
    
    # Gather the logits at the target position for each sample in the batch using a loop
    final_logits = []
    for i in range(full_logits.size(0)):
        final_logits.append(full_logits[i, target_positions[i], :])
    final_logits = torch.stack(final_logits, dim=0)
    
    # Convert logits to probabilities
    final_probs = F.softmax(final_logits, dim=-1)
    
    # Find token IDs for "0" and "1"
    token_0_id = get_token_id("0", tokenizer)
    token_1_id = get_token_id("1", tokenizer)

    metrics["p0"] = final_probs[:, token_0_id].mean().item()
    metrics["p1"] = final_probs[:, token_1_id].mean().item()
    metrics["p0_plus_p1"] = metrics["p0"] + metrics["p1"]

    # Below is mostly for debugging - this can be removed when we're confident in the metrics
    top_k = 10
    avg_probs = final_probs.mean(dim=0)
    top_probs, top_indices = torch.topk(avg_probs, k=top_k)
    
    # More robust token decoding with error handling
    top_tokens = []
    for idx in top_indices:
        try:
            token = tokenizer.decode([idx.item()])
            # Handle potential whitespace/special chars for cleaner display
            if token == '':
                token = '<empty>'
            elif token.isspace():
                token = f'<space:{repr(token)}>'
            top_tokens.append(token)
        except:
            top_tokens.append(f'<decode_error:{idx.item()}>')

    print(f"Top {top_k} tokens and their avg probabilities:")
    for token, prob in zip(top_tokens, top_probs):
        print(f"  '{token}': {prob:.4f}")

    print(f"p(0): {avg_probs[token_0_id]:.4f}")
    print(f"p(1): {avg_probs[token_1_id]:.4f}")
    print(f"p(0) + p(1): {avg_probs[token_0_id] + avg_probs[token_1_id]:.4f}")

    # Optional: Show what tokens 0 and 1 actually decode to for verification
    print(f"Token {token_0_id} decodes to: '{tokenizer.decode([token_0_id])}'")
    print(f"Token {token_1_id} decodes to: '{tokenizer.decode([token_1_id])}'")


def calculate_custom_metrics(
    logits: List[torch.Tensor],
    labels: torch.Tensor,
    tokenizer=None,
    ignore_index: int = -100
) -> Dict[str, float]:
    """
    Calculate custom metrics for the model.
    
    Args:
        logits: Raw model outputs (it's a list because the sequence dimensions gets chunked)
        labels: Real tokens as well as padding (e.g., -100 for ignored tokens)
        tokenizer: Tokenizer to look up token IDs
        ignore_index: Index to ignore in calculations (e.g., -100 for padding)
        
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
        
    # Example (I now strongly recommend this way!): my_metric_function(metrics, logits, labels, tokenizer=tokenizer, ignore_index=ignore_index)
    # The dictionary `metrics` will be updated in-place with new metric names and values so we don't need to return it.
    # You can leave out labels, etc. if you don't need them.
    # Add the function above this one
    calculate_prob_0_1_sum(metrics, logits, labels, tokenizer=tokenizer, ignore_index=ignore_index)
    
    return metrics