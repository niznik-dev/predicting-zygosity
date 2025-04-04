#!/bin/bash
#SBATCH --job-name=mz_test # Job name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G # Memory allocation
#SBATCH --time=0:29:00 # Time limit (HH:MM:SS)
#SBATCH --mail-type=begin # Email when job starts
#SBATCH --mail-type=end # Email when job ends
#SBATCH --mail-user=<NETID>@princeton.edu
#SBATCH --gres=gpu:1 # Request 1 GPU
##SBATCH --account=<ACT>
##SBATCH --partition=<PAR>
##SBATCH --constraint=gpu80

RUN_NAME=""
INPUT_FORMATTING=""

module purge
module load anaconda3/2024.10
conda activate tttestcg

python eval.py /home/$USER/scratch/zyg_out_$RUN_NAME/epoch_0 /home/niznik/scratch/zyg_in/$INPUT_FORMATTING /home/niznik/scratch/zygosity_output_$RUN_NAME/epoch_0