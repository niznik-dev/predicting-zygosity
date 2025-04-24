# predicting-zygosity
A reboot of Alessandra's repo now attached to our new Project, "Everything Predictor"!

# Prerequisites

- Cloning and Pulling from GitHub (properly!)
  - [Version Control with Git](https://swcarpentry.github.io/git-novice/) from Software Carpentry
  - Specifically: Episode 7
- Basic Linux and Shell Commands
  - [The Unix Shell](https://swcarpentry.github.io/shell-novice/) from Software Carpentry
  - Specifically: Episodes 1, 2, 3, and optionally 6
- Working on a Remote Machine
  - A [Digital Ocean Tutorial](https://www.digitalocean.com/community/tutorials/how-to-use-visual-studio-code-for-remote-development-via-the-remote-ssh-plugin) for SSH using VSCode
- Understanding Python package management with conda and pip
  - [Introduction to Conda for (Data) Scientists](https://carpentries-incubator.github.io/introduction-to-conda-for-data-scientists/) from The Carpentries Incubator
  - Specifically: Parts 1 and 2
- [Getting a HuggingFace Account and Token](https://huggingface.co/docs/hub/en/security-tokens)
- Basic Slurm Knowledge
  - We recommend the [Princeton Research Computing](https://researchcomputing.princeton.edu/support/knowledge-base/slurm) primer
- Optional: Experience with Python coding for reading the codebase and/or adding functionality
  - [Programming with Python](https://swcarpentry.github.io/python-novice-inflammation/) from Software Carpentries

# Installation

## With conda/pip (the original way):

Specific to della:
```
ssh user@della-gpu.princeton.edu
module load anaconda3/2024.10
```

All machines with conda and GPU visibility (including della):
```
conda create -n ttenv pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
conda activate ttenv
pip install torchao torchtune wandb
```

Once built, make sure to activate the environment when working on any tune command (and load the anaconda module if you're on della)

## With minimal conda:

Specific to della:
```
ssh user@della-gpu.princeton.edu
module load anaconda3/2024.10
```

All machines with conda and GPU visibility (including della):
```
conda create -n ttenv python=3.12
conda activate ttenv
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip3 install torchao torchtune wandb
```

Once built, make sure to activate the environment when working on any tune command (and load the anaconda module if you're on della)

## If you ever need torchtune's nightly build...

Sometimes you may want to work with a feature that's merged into torchtune but not part of an official release yet. In that case, you can create an environment like above but split the final pip install into these parts instead:
```
pip3 install torchao
pip3 install --pre torchtune --extra-index-url https://download.pytorch.org/whl/nightly/cpu --no-cache-dir
pip3 install wandb
```

A new environment for this is recommended - `ttenv-nightly` is one possible name.

# Downloading a model

Next you'll need a model to finetune and evaluate with. Here's how to get one *the torchtune way*:

```
tune download meta-llama/<model_name> --output-dir <model_dir> --hf-token <hf-token>
```
**model_name**: We've specifically worked with:
* *Llama-2-7b-hf*
* Llama-3.1-8B-Instruct
* Llama-3.2-1B-Instruct (most common)

**model_dir**: A suggestion is `/scratch/gpfs/$USER/torchtune_models/<model_name>` - you'll need this in finetune.yaml later

**hf-token**: You can get this from your HuggingFace account; **NEVER** commit this to a repo!

# Formatting Input Files to JSON

First, obtain the csv files for this project (named in the format twindat_sim_?k_NN.csv, where ? = thousands of rows and NN is either 24 or 99 (variables)) and place them in a nice folder - I suggest `zyg_raw`.

Then run the following (can be done on a login node - only takes a minute or so):

```
python preproc.py /path/to/csv/files
```

/path/to/csv/files: A suggestion is `/scratch/gpfs/$USER/zyg_raw`

When complete, multiple JSON files will be created in the same input folder you specify above; these can then be moved to a folder such as `/scratch/gpfs/$USER/zyg_in` - you'll need this in finetune.yaml as well

# A Test Run

*This will likely change as we complete [#1](https://github.com/niznik-dev/predicting-zygosity/issues/1)...*

**finetune.yaml**: fill in each of the "COMMON CONFIGS TO CHANGE" near the top of the file
* If you followed the suggestions, the dirs can be left alone
* A bach_size of 4 is fine for now

**finetune.slurm**: Fix a few things:
* Replace <NETID> so that the email is valid for you
* If you typically use an account and/or partition in slurm scripts, uncomment those lines with the appropriate settings
* For this test, you can leave the gpu80 constraint commented out
* Change the folder after scratch in the mkdir command to match `my_wandb_run_name` from the yaml (leave the /logs/wandb ending!)
* Update the path to config if you do not follow the listed convention

Then run `sbatch finetune.slurm` and watch the magic happen!



