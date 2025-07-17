# Binary Sequence Dataset Generator

This script generates synthetic binary sequence datasets for use in binary classification tasks. Labels can be assigned either by **bit parity** (even/odd number of 1s) or via a **Bernoulli distribution** with probability `p`.

## Features

* Supports both deterministic parity labeling and probabilistic labeling.
* Option to inject noise when using parity labels (flip labels with probability `p`).
* Outputs two JSON files: `train.json` and `test.json`, formatted for LLM-style instruction tuning.
* Ensures at least one copy of each possible binary sequence of length `n` for varying dataset sizes (set $2^\texttt{bitlength} = N$ if you don't want repetition). 

## Usage

To create a parity dataset (no noise) with 2^5=32 unique sequences we could have

```
python generate.py --bit_length 5 --N 1000000 --p 0 --bit_parity True --test_size 1000
```

To create a probabilistic dataset with $p = 0.3$ we could have

```
python generate.py --bit_length 8 --N 500000 --p 0.5 --bit_parity False --test_size 1000
```

## Arguments

* `--bit_length`: Length of each binary sequence (e.g., 5).
* `--N`: Total number of examples (must be ≥ 2^bit\_length).
* `--test_size`: Number of test examples.
* `--bit_parity`: `True` for parity-based labeling, `False` for probabilistic.
* `--p`: Noise level (if parity) or Bernoulli probability (if probabilistic).
* `--seed`: Random seed (default: 42).
* `--outdir`: Output directory (default: current directory).

## Output Format

Each `.json` file contains a list of examples in this format:

```json
{
  "instruction": "",
  "input": "01011",
  "output": "1"
}
```

## How to integrate it with the codebase

Generate train.json and test.json into your current working directory and then use `generate_slurm_script.py` with the following arguments:

```
python generate_slurm_script.py \
  --my_wandb_project predictable_or_not \
  --my_wandb_run_name binary_sequence_test \
  --input_dir_base /home/niznik/scratch/GitHub/cruijff-kit/tests/bit_sequences/ \
  --input_formatting '' \
  --dataset_filename train.json \
  --dataset_val_filename test.json \
  --train_on_input true \
  --batch_size 1 \
  --epochs 10 \
  --save_adapter_weights_only true \
  --log_every_n_steps 1 \
  --run_val_every_n_steps 4 \
  --conda_env ttenv-nightly \
  --custom_recipe lora_finetune_single_device_val.py \
  --account msalganik \
  --constraint gpu80

sbatch finetune_filled.slurm
```
