import argparse
import os
import yaml

# Skip these when writing the yaml file
SLURM_ONLY = ['time', 'gpus', 'conda_env', 'account', 'partition', 'constraint']

parser = argparse.ArgumentParser()

# ----- Required YAML Args Reused in Templating -----
parser.add_argument("--my_wandb_project", type=str, default="PredictingZygosity", help="Project for when results are synced to wandb")
parser.add_argument("--my_wandb_run_name", type=str, default="MyFirstRun", help="Name for when results are synced to wandb")
parser.add_argument("--input_formatting", type=str, default="raw", help="Name of the folder where your input files are stored within input_dir; useful for multiple formatting styles (e.g. difference vs raw values)")

parser.add_argument("--output_dir_base", type=str, default="/home/$USER/scratch/", help="Full path to the output file folders (final output folder will be 'zyg_out_' + my_wandb_name within this folder)")
parser.add_argument("--input_dir_base", type=str, default="/home/$USER/scratch/zyg_in/", help="Full path to the input file folders")
parser.add_argument("--models_dir", type=str, default="/home/$USER/scratch/torchtune_models/", help="Full path to the model file folders")

# ----- Optional YAML Args -----
parser.add_argument("--max_steps_per_epoch", type=int, help="Maximum steps per epoch (useful for debugging)")
parser.add_argument("--run_val_every_n_steps", type=int, help="Number of epochs to train for")

# ------ Slurm Args -----
parser.add_argument("--time", type=str, default="00:15:00", help="Time to run the job (HH:MM:SS)")
parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
parser.add_argument("--conda_env", type=str, default="ttenv", help="Name of the conda environment to use")

parser.add_argument("--account", type=str, help="Slurm account to use")
parser.add_argument("--partition", type=str, help="Slurm partition to use")
parser.add_argument("--constraint", type=str, help="Slurm constraint to use")

args = parser.parse_args()

username = os.environ.get("USER")

# First edit the yaml template
with open("templates/finetune_template.yaml", "r") as f:
    config = yaml.safe_load(f)

for key, value in vars(args).items():
    if key in SLURM_ONLY:
        continue
    # Special cases first
    elif key == "input_dir_base":
        config["input_dir"] = value + args.input_formatting + "/"
    elif key == "output_dir_base":
        full_output_dir = value + "zyg_out_" + args.my_wandb_run_name + "/"
        config["output_dir"] = full_output_dir
    # The rest are straightforward
    else:
        config[key] = value

for key in ['input_dir', 'output_dir', 'models_dir']:
    config[key] = config[key].replace("$USER", username)

with open("finetune_filled.yaml", "w") as f:
    yaml.dump(config, f, sort_keys=False)

# Now create the slurm script
with open("templates/finetune_template.slurm", "r") as f:
    slurm_script = f.read()

slurm_script = slurm_script.replace("<JOBNAME>", args.my_wandb_run_name)
# TODO - lookup reasonable memory/time values based on model choice (create a table somewhere)
slurm_script = slurm_script.replace("00:15:00", args.time)
slurm_script = slurm_script.replace("<NETID>", username)

if args.gpus > 1:
    slurm_script = slurm_script.replace("#SBATCH --cpus-per-task=1", "#SBATCH --cpus-per-task=" + str(args.gpus))
    slurm_script = slurm_script.replace("#SBATCH --gres=gpu:1", "#SBATCH --gres=gpu:" + str(args.gpus))
    slurm_script = slurm_script.replace("lora_finetune_single_device", "--nproc_per_node=" + str(args.gpus) + " lora_finetune_distributed")
if args.account:
    slurm_script = slurm_script.replace("##SBATCH --account=<ACT>", "#SBATCH --account=" + args.account)
if args.partition:
    slurm_script = slurm_script.replace("##SBATCH --partition=<PART>", "#SBATCH --partition=" + args.partition)
if args.constraint:
    slurm_script = slurm_script.replace("##SBATCH --constraint=<CONST>", "#SBATCH --constraint=" + args.constraint)

slurm_script = slurm_script.replace("<CONDA_ENV>", args.conda_env)
slurm_script = slurm_script.replace("<OUTPUT_DIR>", full_output_dir)
slurm_script = slurm_script.replace("$USER", username)

with open("finetune_filled.slurm", "w") as f:
    f.write(slurm_script)