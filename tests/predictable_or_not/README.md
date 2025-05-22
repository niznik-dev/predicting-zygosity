# Predictable or Not Test

## Purpose

To run a simple fine-tuning task on small datasets to see how loss (in both fine-tuning and validation) differs between considering just the output (default) vs both input *and* output. The second is accomplished by adding the following to the yaml:

```
dataset:
  train_on_input: true
```

There are four scenarios:
1. Predictable Input and Predictable Output (e.g. input is '42,43,44,45,46,' and output is '47' - a clear pattern)
1. Predictable Input and Unpredictable Output (e.g. input is '42,43,44,45,46' and output is a random integer (1-1000))
1. Unpredictable Input and Predictable Output (e.g. input is '1,1000,400,300,700' and output is always '42')
1. Unpredictable Input and Unpredictable Output (e.g. everything is random)

## How to Run

### Part 0 - Consider Changing...

- input_dir_base: make sure this points to your copy of the repo
- dataset_split_point: fine-tuning will be on this percent of the dataset and the rest will be used for validation; you can change this to 80 or whatever value you think is appropriate
- account: can be omitted unless you know you have multiple accounts (make sure to remove the final \ after the conda_env line then!)

### Part 1 - Finetune on Output Only

Use `generate_slurm_script.py` with the following arguments:

(Note: you'll have to run this four times with different combinations of predictable and unpredictable: pp, pu, up, and uu)

```bash
python generate_slurm_script.py \
  --my_wandb_project predictable_or_not \
  --my_wandb_run_name pp-false-test \
  --input_dir_base /home/niznik/scratch/GitHub/predicting-zygosity/tests/predictable_or_not/ \
  --input_formatting '' \
  --dataset_filename pp.json \
  --dataset_val_filename pp.json \
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

(Note: you'll have to run this four times with different combinations of predictable and unpredictable: pp, pu, up, and uu)

```bash
python generate_slurm_script.py \
  --my_wandb_project predictable_or_not \
  --my_wandb_run_name pp-true-test \
  --input_dir_base /home/niznik/scratch/GitHub/predicting-zygosity/tests/predictable_or_not/ \
  --input_formatting '' \
  --dataset_filename pp.json \
  --dataset_val_filename pp.json \
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