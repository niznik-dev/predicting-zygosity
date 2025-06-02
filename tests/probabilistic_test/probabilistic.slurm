#!/bin/bash
#SBATCH --job-name=p-sweep
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --array=0-10                   # 11 tasks: 0 → 10
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=02:30:00
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=ar0241@princeton.edu
#SBATCH --output=logs/p-sweep_%a.out
#SBATCH --error=logs/p-sweep_%a.err

module purge
module load anaconda3/2024.6
conda activate ttenv

# define the grid
p_list=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
p=${p_list[$SLURM_ARRAY_TASK_ID]}

# per‐run directories
RUNDIR=scratch/stochastic/run_p${p}
mkdir -p ${RUNDIR}

# 1) generate
~/.conda/envs/ttenv/bin/python3.12 scratch/stochastic/generate_test.py \
    --n 14 \
    --p ${p} \
    --test_size 100 \
    --outdir ${RUNDIR}

# 2) finetune 
~/.conda/envs/ttenv/bin/python3.12 -m torchtune._cli.tune run lora_finetune_single_device \
    --config scratch/stochastic/train_test.yaml \
    epochs=20 \
    input_dir=${RUNDIR} \
    output_dir=/home/ar0241/scratch/twins/output_p${p}


# 3) eval (evaluating on the last epoch, change folder name)
export RESULTS_CSV=/home/ar0241/scratch/twins/results.csv  # (optional)
~/.conda/envs/ttenv/bin/python3.12 scratch/stochastic/eval_test.py \
    --base_model_path /home/ar0241/scratch/torchtune_models/Llama-3.2-1B-Instruct \
    --adapter_path  /home/ar0241/scratch/twins/output_p${p}/epoch_19 \
    --eval_file      ${RUNDIR}/prob_train.json \
    --true_p         ${p}


~/.conda/envs/ttenv/bin/python3.12 scratch/stochastic/stochastic_plot.py --n 14 --test_size 100 --eval_on_test TRUE --epochs 20
