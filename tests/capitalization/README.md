# Capitalization Test

## Purpose

To run a simple fine-tuning task on a small dataset of five-letter words and their capitalized versions to see if the pattern will be followed when words of other lengths are given to the fine-tuned model.

## How to Run

### Part 0 - Consider Changing...

- input_dir_base: make sure this points to your copy of the repo
- account: can be omitted unless you know you have multiple accounts (make sure to remove the final \ after the conda_env line then!)

### Part 1 - Finetune the Model

Use `generate_slurm_script.py` with the following arguments:

```bash
python generate_slurm_script.py \
  --my_wandb_project capitalization \
  --my_wandb_run_name finetune-five \
  --input_dir_base /home/niznik/scratch/GitHub/predicting-zygosity/tests/capitalization/ \
  --input_formatting '' \
  --dataset_filename fiveLetterCapitalization.json \
  --dataset_val_filename fiveLetterCapitalization.json \
  --dataset_split_point 100 \
  --batch_size 1 \
  --epochs 1 \
  --log_every_n_steps 1 \
  --run_val_every_n_steps 10000 \
  --conda_env ttenv-nightly \
  --custom_recipe lora_finetune_single_device_val.py \
  --account msalganik \
  --constraint gpu80

sbatch finetune_filled.slurm
```

(Currently, we need to extract all of the validation related things from the yaml file - will be addressed in [#49](https://github.com/niznik-dev/predicting-zygosity/issues/49))

### Part 2 - Upload to Weights & Biases

Run the following to upload your run:

```bash
wandb sync /path/to/output/folder/logs/wandb/latest-run
```

### Part 3 - Test the model

```
sbatch eval.slurm
```

and examine the slurm log file for the output. (Currently, we're seeing very low probability for the capitalized words of length 5 *or* 6 but it does increase from ~0.01% to ~0.2% for some words)