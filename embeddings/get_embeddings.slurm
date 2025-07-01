#!/bin/bash
#SBATCH --job-name=get_embed # Job name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G # Memory allocation
#SBATCH --time=30:00:00 # Time limit (HH:MM:SS)
#SBATCH --mail-type=begin # Email when job starts
#SBATCH --mail-type=end # Email when job ends
#SBATCH --mail-user=drigobon@princeton.edu
#SBATCH --gres=gpu:1 # Request 1 GPU
##SBATCH --account=<ACT>
##SBATCH --partition=<PART>
##SBATCH --constraint=<CONST>
#SBATCH --output=/home/drigobon/scratch/zyg-out/100k-20epoch/get_embeddings.out



module purge
module load anaconda3/2024.10
conda activate ttenv

# ! Magic Numbers & Directories
NUM_OBS=20000 # 200 = 1% of 20k dataset
MAX_EPOCH=20 # for iteration through fine-tuned model at checkpoints. 

# Directories
RUN_NAME="100k-20epoch" # Name of the run, used for embeddings input and figure output directories
# This should go into name of the output file of SBATCH...

BASE_DIR="/home/drigobon/scratch/"
PYTHON_FILE_PATH="$BASE_DIR/embeddings-analysis/get_embeddings.py"
BASE_MODEL_PATH="$BASE_DIR/torchtune_models/Llama-3.2-1B-Instruct"
EVAL_DATA_PATH="$BASE_DIR/zyg-in/ptwindat_eval.json"
ADAPTER_PATH="$BASE_DIR/zyg-out/$RUN_NAME/"
EMBED_SAVE_PATH="$BASE_DIR/zyg-out/$RUN_NAME/embeddings/"

# ! End Magic Numbers & Directories

# Create output directory if it doesn't exist
mkdir -p "$EMBED_SAVE_PATH"

# Get embeddings for base_model (no loop needed)
python3 "$PYTHON_FILE_PATH" \
    --base_model_path "$BASE_MODEL_PATH" \
    --eval_file "$EVAL_DATA_PATH" \
    --output_embed "$EMBED_SAVE_PATH/base.json" \
    --num_obs "$NUM_OBS"


# Get embeddings for fine-tuned model (include --adapter_path)
#   From epoch 0 to MAX_EPOCH
#   Finetuned models must be in ADAPTER_PATH/epoch_X/...
#   Saves files to EMBED_SAVE_PATH/epoch_X.json
# ! NOTE! This assumes that EVERY epoch we get a new checkpoint...
for ((i=0; i<MAX_EPOCH; i++)); do

    python3 "$PYTHON_FILE_PATH" \
        --base_model_path "$BASE_MODEL_PATH" \
        --eval_file "$EVAL_DATA_PATH" \
        --output_embed "$EMBED_SAVE_PATH/epoch_$i.json" \
        --num_obs "$NUM_OBS" \
        --adapter_path "$ADAPTER_PATH/epoch_$i"

done







