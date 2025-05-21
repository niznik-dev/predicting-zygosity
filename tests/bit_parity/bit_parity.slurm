#!/bin/bash
#SBATCH --job-name=bit_parity_test  # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --mem=64G                      # Memory allocation
#SBATCH --time=00:10:00                # Time limit (HH:MM:SS)
#SBATCH --mail-type=begin              # Email when job starts
#SBATCH --mail-type=end                # Email when job ends
#SBATCH --mail-user=ar0241@princeton.edu
#SBATCH --gres=gpu:1                    # Request 1 GPU

module purge
module load anaconda3/2024.6
conda activate ttenv
~/.conda/envs/ttenv/bin/python3.12 scratch/bit/generate_test.py
~/.conda/envs/ttenv/bin/python3.12 -m torchtune._cli.tune \
    run lora_finetune_single_device \
    --config scratch/bit/train_test.yaml epochs=1
~/.conda/envs/ttenv/bin/python3.12 scratch/stochastic/eval_test.py
