# Simple tests of whether this model was finetuned

This utility script checks whether a fine-tuned language model using PEFT adapters is meaningfully different from its base model. It does so by computing two metrics — **perplexity** and **logit mean squared error (MSE)** — on a user-provided input prompt.

---

# How to run it?

```
python is_this_a_finetuned_model.py \
  --base_model_path /path/to/base_model \
  --adapter_path /path/to/adapter_checkpoint \
  --prompt "Add prompt here" \
  --ppl_threshold 0.01 \
  --mse_threshold 0.0001
```

# Example

```
python is_this_a_finetuned_model.py   --base_model_path /home/ar0241/scratch/torchtune_models/Llama-3.2-1B-Instruct   --adapter_path /home/ar0241/scratch/twins/output_p0/epoch_9 --prompt "The output of 0001011 is 1" --ppl_threshold 0.01  --mse_threshold 0.0001
```

# Example output 

```
=== Model Comparison ===
Input: The output of 0001011 is 1
Base Model Perplexity:    67.575
Adapter Model Perplexity: 67.129
Logit MSE Difference:     0.003712
✅ Adapter model is different — likely fine-tuned.
```

# Explanation

This script performs a diagnostic comparison between a pretrained language model $M_{\text{base}}$ and a fine-tuned derivative $M_{\text{adapter}}$ built using **parameter-efficient fine-tuning (PEFT)**. The goal is to evaluate whether the adapter has non-trivially altered the functional behavior of the model.

### Token-level likelihood and perplexity

We begin by tokenizing an arbitrary prompt provided at the command line. Both $M_{\text{base}}$ and $M_{\text{adapter}}$ generate logits $\mathbf{z} \in \mathbb{R}^{T \times V}$ over a vocabulary of size $V$ for each of $T$ tokens. These logits are used to compute a **next-token prediction loss** via cross-entropy, averaged across time steps. The exponential of this loss gives **perplexity**, a standard metric indicating how "confident" the model is in generating the observed text. If fine-tuning is effective, we expect the perplexity of the adapter to be lower than that of the base model, especially when the prompt reflects the fine-tuning domain (depending on how many epochs were run and on the specific task).

### Logit space divergence

Then, this computes the **mean squared error (MSE)** between the raw logits of the two models. Because logits represent the pre-softmax activations over vocabulary tokens, they encode the model's inductive biases before normalization. A small MSE implies that the adapter model is functionally near-identical to the base, suggesting the adapter is not having any measurable effect. This metric is sensitive to even minor deviations in model behavior.

### Summary

Together, these metrics form a complementary diagnostic:

* **Perplexity** captures model confidence over the actual sequence.
* **Logit MSE** captures the degree of deviation in token-level preferences across the vocabulary space.

The combination allows you to distinguish between (1) models that learned nothing (identical logits and perplexity), (2) models that diverged without gaining fluency (higher perplexity but diverging logits), and (3) successful adapters (lower perplexity and high logit deviation).





  


  

