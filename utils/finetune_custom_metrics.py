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


def calculate_custom_metrics(
    logits: List[torch.Tensor],
    labels: torch.Tensor,
    tokenizer=None,
    ignore_index: int = -100
) -> Dict[str, torch.Tensor]:
    """
    Calculate custom metrics for the model.
    
    Args:
        logits: Raw model outputs (it's a list because the sequence dimensions gets chunked)
        labels: Ground truth labels (to be used in MSE calculation)
        tokenizer: Tokenizer to look up token IDs
        ignore_index: Index to ignore in calculations (e.g., -100 for padding)
        
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
        
    # TODO - first person to add one here wins!
    # Example (I now strongly recommend this way!): my_metric_function(metrics, logits, labels, tokenizer=tokenizer, ignore_index=ignore_index)
    # The dictionary `metrics` will be updated in-place with new metric names and values so we don't need to return it.
    # You can leave out labels, etc. if you don't need them.
    # Add the function above this one
    
    return metrics