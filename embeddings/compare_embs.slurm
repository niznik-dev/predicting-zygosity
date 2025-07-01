#!/bin/bash
#SBATCH --job-name=compare_embs # Job name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --ntasks=1 # Number of tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G # Memory allocation
#SBATCH --time=11:59:00 # Time limit (HH:MM:SS)
#SBATCH --mail-type=begin # Email when job starts
#SBATCH --mail-type=end # Email when job ends
#SBATCH --mail-user=drigobon@princeton.edu
#SBATCH --output=/home/drigobon/scratch/zyg-out/100k-20epoch/compare_embs.out


module purge
module load anaconda3/2024.10
conda activate ttenv-umap # Note that env depends on emb_type specified in the python header


# ! Magic Numbers & Directories

# Embeddings
EMB_TYPE='PCA'
    # Notes: 
    #   - Options: 'PCA', 'TSNE', 'UMAP'
    #   - If is not 'PCA', then no cos_sim plots produced.
    #   - If is either: 'PCA' or 'UMAP' then 'fit_to_base' parameter is enabled
    #   - Other things missing...?
FIT_TO_BASE=0
    # Notes: For plotting
    #   - If True(=1), then PCA/UMAP will fit to the base model's embedding space
    #   - If False(=0), then PCA/UMAP will fit to the fine-tuned model's embedding space
    #   - Only applies if EMB_TYPE is either 'PCA' or 'UMAP'
N_COMPONENTS=2 # Number of components for PCA/UMAP/TSNE
    # Notes:
    #   - Must be >=2 for 2d plots, >=3 for 3d plots

# Directories
RUN_NAME="100k-20epoch" # Name of the run, used for embeddings input and figure output directories
# This should go into name of the output file of SBATCH...

BASE_DIR="/home/drigobon/scratch/"
EMBEDDINGS_DIR="$BASE_DIR/zyg-out/$RUN_NAME/embeddings/"
FIG_DIR_BASE="$BASE_DIR/zyg-out/$RUN_NAME/figs-$EMB_TYPE/"
PATTERN="epoch_*.json" # pattern in EMBEDDINGS_DIR to match embeddings files from model checkpoints (does NOT include base model)

EVAL_FILE_PATH="$BASE_DIR/zyg-in/ptwindat_eval.json"
PYTHON_FILE_PATH="$BASE_DIR/embeddings-analysis/compare_embs.py"

# Data
NUM_OBS=20000 # (200 = 1% of 20k dataset for zyg synthetic validation set)

# Plotting
SAVE_PLOTS_2D=1 # whether to save the 2d plots or not

SAVE_PLOTS_3D=0 # whether to save the 3d plots or not

SAVE_PLOTS_GRID=1 # whether to save the 2d plots in a grid or not
K_EPOCH=5 # how many epochs to plot in a grid (only matters if SAVE_PLOTS_GRID=1)

SAVE_PLOTS_COS_SIM=0 # whether to save the cosine similarity plots or not (only matters for emb_type='PCA')
MAX_PC_TO_COMPARE=2 # which principal component we're comparing (cannot be greater than n_components)

# ! End Magic Numbers & Directories


# Directories made in the python script, so no need to create them here

# Run script
python3 "$PYTHON_FILE_PATH" \
    --emb_type "$EMB_TYPE" \
    --fit_to_base "$FIT_TO_BASE" \
    --n_components "$N_COMPONENTS" \
    --embeddings_dir "$EMBEDDINGS_DIR" \
    --pattern "$PATTERN" \
    --fig_dir_base "$FIG_DIR_BASE" \
    --eval_file "$EVAL_FILE_PATH" \
    --num_obs "$NUM_OBS" \
    --save_plots_2d "$SAVE_PLOTS_2D" \
    --save_plots_3d "$SAVE_PLOTS_3D" \
    --save_plots_grid "$SAVE_PLOTS_GRID" \
    --k_epoch "$K_EPOCH" \
    --save_plots_cos_sim "$SAVE_PLOTS_COS_SIM" \
    --max_pc_to_compare "$MAX_PC_TO_COMPARE" \




