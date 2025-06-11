"""
Custom metrics for model evaluation during fine-tuning
"""

import torch
import torch.nn.functional as F
from typing import Dict


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
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Token '{token}' not found in tokenizer vocabulary")
        return token_ids[0]
    except Exception as e:
        raise ValueError(f"Failed to encode token '{token}': {e}")


def calculate_mse_from_logits(
    logits: torch.Tensor, 
    labels: torch.Tensor, 
    tokenizer=None,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Calculate MSE between probability of class 1 and binary labels.
    
    Args:
        logits: Raw model outputs [batch_size * seq_len, vocab_size]
        labels: Ground truth labels [batch_size * seq_len]
        tokenizer: Tokenizer to look up token ID for "1"
        ignore_index: Index to ignore in loss calculation
        
    Returns:
        MSE loss as tensor
    """
    # Filter out ignored labels
    valid_mask = labels != ignore_index
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device)
    
    valid_logits = logits[valid_mask]
    valid_labels = labels[valid_mask]
    
    # Convert logits to probabilities
    probs = F.softmax(valid_logits, dim=-1)
    
    # Find token ID for "1"
    token_1_id = get_token_id("1", tokenizer)
    
    # Extract probability of token "1"
    prob_1 = probs[:, token_1_id]
    
    # Convert labels to float for MSE calculation
    binary_labels = valid_labels.float()
    
    # Calculate MSE
    mse = F.mse_loss(prob_1, binary_labels, reduction='mean')
    
    return mse


def get_third_most_likely_token_info(logits: torch.Tensor) -> tuple[int, float]:
    """
    Get the third most likely token ID and its probability for debugging.
    Uses first valid sample from the batch.
    
    Args:
        logits: Raw model outputs [batch_size, vocab_size]
        
    Returns:
        Tuple of (token_id, probability)
    """
    # Get probabilities for first sample
    probs = F.softmax(logits[0], dim=-1)
    
    # Get top 3 token indices and probabilities
    top_probs, top_indices = torch.topk(probs, k=3, dim=-1)
    
    # Return third most likely (index 2)
    return top_indices[2].item(), top_probs[2].item()


def calculate_prob_0_1_sum(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer=None,
    ignore_index: int = -100
) -> torch.Tensor:
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
    # Filter out ignored labels
    valid_mask = labels != ignore_index
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device)
    
    valid_logits = logits[valid_mask]
    
    # Convert logits to probabilities
    probs = F.softmax(valid_logits, dim=-1)
    
    # Find token IDs for "0" and "1"
    token_0_id = get_token_id("0", tokenizer)
    token_1_id = get_token_id("1", tokenizer)
    
    # Sum p(0) + p(1) for each valid token
    prob_0_1_sum = probs[:, token_0_id] + probs[:, token_1_id]
    
    # Return average across all valid tokens
    return prob_0_1_sum.mean()


def calculate_custom_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tokenizer=None,
    ignore_index: int = -100
) -> Dict[str, torch.Tensor]:
    """
    Calculate custom metrics for the model.
    
    Args:
        logits: Raw model outputs
        labels: Ground truth labels
        tokenizer: Tokenizer to look up token IDs
        ignore_index: Index to ignore in calculations
        
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    
    # MSE metric
    metrics['mse'] = calculate_mse_from_logits(logits, labels, tokenizer, ignore_index)
    
    # Average sum of p(0) + p(1) 
    metrics['avg_prob_0_1_sum'] = calculate_prob_0_1_sum(logits, labels, tokenizer, ignore_index)
    
    # Debug metrics: third most likely token and its probability
    if logits.numel() > 0:
        third_token, third_prob = get_third_most_likely_token_info(logits)
        metrics['third_most_likely_token'] = torch.tensor(float(third_token))
        metrics['third_token_prob'] = torch.tensor(third_prob)
    
    return metrics