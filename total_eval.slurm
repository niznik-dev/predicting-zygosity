#!/bin/bash
#SBATCH --job-name=cr_total_eval # Job name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G # Memory allocation
#SBATCH --time=00:15:00 # Time limit (HH:MM:SS)
#SBATCH --mail-type=begin # Email when job starts
#SBATCH --mail-type=end # Email when job ends
#SBATCH --mail-user=<NETID>@princeton.edu
#SBATCH --gres=gpu:1 # Request 1 GPU
##SBATCH --array=0-<ARRAY_MAX> # Uncomment if using job arrays
##SBATCH --account=<ACT>
##SBATCH --partition=<PART>
##SBATCH --constraint=<CONST>

module purge
module load anaconda3/2024.10
conda activate ttenv

python total_eval.py \
  --model_path /scratch/gpfs/$USER/torchtune_models/Llama-3.2-1B-Instruct/ \
  --input_prefix /scratch/gpfs/$USER/zyg_in/ \
  --input_filename val.json \
  --adapter_path /scratch/gpfs/$USER/<RUN_NAME>/epoch_0/ \
  --batch_size 8