#!/bin/bash
#SBATCH --job-name=save_hidden
#SBATCH --output=save_hidden.out
#SBATCH --error=save_hidden.err
#SBATCH --time=13:45:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

# Optional: specify GPU type
# SBATCH --constraint=a100

cd /scratch/gpfs/TROYANSKAYA/sokolova/predicting-zygosity/embeddings/
# Load environment
module load anaconda3/2024.10
source /scratch/gpfs/TROYANSKAYA/sokolova/predicting-zygosity/ttenv/bin/activate

# Run your script
python save_all_hidden.py
