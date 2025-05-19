# Input Training Test

## Purpose

To run a simple fine-tuning task on a small dataset to see how loss (in both fine-tuning and validation) differs between considering just the output (default) vs both input *and* output. The second is accomplished by adding the following to the yaml:

```
dataset:
  train_on_input: true
```

## How to Run

### Part 0 - Consider Changing...

- input_dir_base: make sure this points to your copy of the repo
- dataset_split_point: fine-tuning will be on this percent of the dataset and the rest will be used for validation; you can change this to 80 or whatever value you think is appropriate
- account: can be omitted unless you know you have multiple accounts (make sure to remove the final \ after the conda_env line then!)

### Part 1 - Finetune on Output Only

Use `generate_slurm_script.py` with the following arguments:

```bash
python generate_slurm_script.py \
  --my_wandb_project input_training_tests \
  --my_wandb_run_name false-test \
  --input_dir_base /home/niznik/scratch/GitHub/predicting-zygosity/tests/input_training/ \
  --input_formatting '' \
  --dataset_filename simpleCapitalization.json \
  --dataset_val_filename simpleCapitalization.json \
  --dataset_split_point 90 \
  --batch_size 1 \
  --epochs 10 \
  --save_adapter_weights_only true \
  --log_every_n_steps 1 \
  --run_val_every_n_steps 4 \
  --conda_env ttenv-nightly \
  --custom_recipe lora_finetune_single_device_val.py \
  --account msalganik

sbatch finetune_filled.slurm
```

### Part 2 - Finetune on Input AND Output

```bash
python generate_slurm_script.py \
  --my_wandb_project input_training_tests \
  --my_wandb_run_name true-test \
  --input_dir_base /home/niznik/scratch/GitHub/predicting-zygosity/tests/input_training/ \
  --input_formatting '' \
  --dataset_filename simpleCapitalization.json \
  --dataset_val_filename simpleCapitalization.json \
  --train_on_input true \
  --dataset_split_point 90 \
  --batch_size 1 \
  --epochs 10 \
  --save_adapter_weights_only true \
  --log_every_n_steps 1 \
  --run_val_every_n_steps 4 \
  --conda_env ttenv-nightly \
  --custom_recipe lora_finetune_single_device_val.py \
  --account msalganik

sbatch finetune_filled.slurm
```

### Part 3 - Upload to Weights & Biases

Run the following twice (once for each of the above fine-tunings)

```bash
wandb sync /path/to/output/folder/logs/wandb/latest-run
```