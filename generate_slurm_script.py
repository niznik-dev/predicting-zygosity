import argparse
import os
import yaml

parser = argparse.ArgumentParser()

parser.add_argument("--my_wandb_project", type=str, default="PredictingZygosity", help="Project for when results are synced to wandb")
parser.add_argument("--my_wandb_run_name", type=str, default="MyFirstRun", help="Name for when results are synced to wandb")
parser.add_argument("--input_formatting", type=str, default="raw", help="Name of the folder where your input files are stored within input_dir; useful for multiple formatting styles (e.g. difference vs raw values)")

parser.add_argument("--output_dir_base", type=str, default="/home/$USER/scratch/", help="Full path to the output file folders (final output folder will be 'zyg_out_' + my_wandb_name within this folder)")
parser.add_argument("--input_dir_base", type=str, default="/home/$USER/scratch/zyg_in/", help="Full path to the input file folders")
parser.add_argument("--models_dir", type=str, default="/home/$USER/scratch/torchtune_models/", help="Full path to the model file folders")

parser.add_argument("--time", type=str, default="00:15:00", help="Time to run the job (HH:MM:SS)")

parser.add_argument("--account", type=str, help="Slurm account to use")
parser.add_argument("--partition", type=str, help="Slurm partition to use")
parser.add_argument("--constraint", type=str, help="Slurm constraint to use")

args = parser.parse_args()

# First edit the yaml template
with open("templates/finetune_template.yaml", "r") as f:
    config = yaml.safe_load(f)

for key, value in vars(args).items():
    # Special cases first
    if key == "input_dir_base":
        config["input_dir"] = value + args.input_formatting + "/"
    elif key == "output_dir_base":
        full_output_dir = value + "zyg_out_" + args.my_wandb_run_name + "/"
        config["output_dir"] = full_output_dir
    # The rest are straightforward
    else:
        config[key] = value

with open("finetune_filled.yaml", "w") as f:
    yaml.dump(config, f)

# Now create the slurm script
with open("templates/finetune_template.slurm", "r") as f:
    slurm_script = f.read()

slurm_script = slurm_script.replace("<JOBNAME>", args.my_wandb_run_name)
# TODO - lookup reasonable memory/time values based on model choice (create a table somewhere)
slurm_script = slurm_script.replace("00:15:00", args.time)
slurm_script = slurm_script.replace("<NETID>", os.environ.get("USER"))

if args.account:
    slurm_script = slurm_script.replace("##SBATCH --account=<ACT>", "#SBATCH --account=" + args.account)
if args.partition:
    slurm_script = slurm_script.replace("##SBATCH --partition=<PART>", "#SBATCH --partition=" + args.partition)
if args.constraint:
    slurm_script = slurm_script.replace("##SBATCH --constraint=<CONST>", "#SBATCH --constraint=" + args.constraint)

slurm_script = slurm_script.replace("<OUTPUT_DIR>", full_output_dir)

with open("finetune_filled.slurm", "w") as f:
    f.write(slurm_script)