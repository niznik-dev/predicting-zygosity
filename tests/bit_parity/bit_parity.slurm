#!/bin/bash
#SBATCH --job-name=p-sweep
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --array=0-10                   # 11 tasks: 0 → 10
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=0:30:00
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=ar0241@princeton.edu
#SBATCH --output=logs/p-sweep_%a.out
#SBATCH --error=logs/p-sweep_%a.err

module purge
module load anaconda3/2024.6
conda activate ttenv

noise_list=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
noise_p=${noise_list[$SLURM_ARRAY_TASK_ID]}

# directories for this run
BASE=/home/ar0241/scratch/bit_parity
RUNDIR=${BASE}/run_noise${noise_p}
OUTDIR=/home/ar0241/scratch/bit_parity/output_noise${noise_p}
RESULTS_CSV=/home/ar0241/scratch/bit_parity/results.csv

mkdir -p ${RUNDIR} ${OUTDIR} $(dirname ${RESULTS_CSV}) logs

echo "=== Task $SLURM_ARRAY_TASK_ID: noise_p=${noise_p} ==="

# 1) generate
~/.conda/envs/ttenv/bin/python3.12 scratch/bit_parity/generate_parity.py \
  --n 14 \
  --test_size 1000 \
  --noise_p ${noise_p} \
  --seed 42 \


# 2) finetune (override input/output dirs via CLI if needed)
~/.conda/envs/ttenv/bin/python3.12 -m torchtune._cli.tune run lora_finetune_single_device \
    --config scratch/bit_parity/train_parity.yaml \
  epochs=20

# 3) eval (change the folder to be the one of the last epoch)
~/.conda/envs/ttenv/bin/python3.12 scratch/bit_parity/eval_parity.py \
  --model_path /home/ar0241/scratch/twins/output_mar25/epoch_19 \
  --eval_file  /home/ar0241/scratch/bit_parity/parity_test.json \
  --noise_p ${noise_p} \
  --results_csv /home/ar0241/scratch/bit_parity/results.csv


~/.conda/envs/ttenv/bin/python3.12 scratch/bit_parity/plot_accuracy.py --n 14 --epochs 20 --test_size 1000

~/.conda/envs/ttenv/bin/python3.12 scratch/bit_parity/check_loss-13.py --n 14 --epochs 20
