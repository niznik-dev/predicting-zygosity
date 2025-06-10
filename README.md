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

## Current recommended instructions

Specific to della:
```
ssh user@della-gpu.princeton.edu
module load anaconda3/2024.10
```

All machines with conda and GPU visibility (including della):
```
conda create -n ttenv python=3.12
conda activate ttenv
pip3 install torch --index-url https://download.pytorch.org/whl/cu126
pip3 install transformers scikit-learn matplotlib # These are only used for eval.py
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

### Current use cases

* Working with val_loss (validation loss) which is not yet in a regular release

# Downloading a model

Next you'll need a model to finetune and evaluate with. Here's how to get one *the torchtune way*:

## Step 1 - Request Access on HuggingFace Website (if necessary)

For Meta models in particular, you'll need to navigate to the model on the HuggingFace website, log in, and agree to their Community License Agreement. Once you have an email confirming that you have been granted access, you can continue to the next step.

For Meta, you can typically follow a URL like this: https://huggingface.co/meta-llama/<model_name> (see options below)

## Step 2 - Run the Command

```
tune download meta-llama/<model_name> --output-dir <model_dir> --hf-token <hf-token>
```
**model_name**: We've specifically worked with:
* Llama-2-7b-hf
* Llama-3.1-8B-Instruct
* Llama-3.2-1B-Instruct (most common)
* Llama-3.3-70B-Instruct

**model_dir**: A suggestion is `/scratch/gpfs/$USER/torchtune_models/<model_name>` - you'll need this in finetune.yaml later

**hf-token**: You can get this from your HuggingFace account; **NEVER** commit this to a repo!

# Formatting Input Files to JSON

## With capitalization dataset (tests/input_training)

The file already exists for you - you'll just need to reference it when running the generator in the next step!

## With twin dataset

First, obtain the csv files for this project (named in the format twindat_sim_?k_NN.csv, where ? = thousands of rows and NN is either 24 or 99 (variables)) and place them in a nice folder - I suggest `zyg_raw`.

Then run the following (can be done on a login node - only takes a minute or so):

```
python preproc.py /path/to/csv/files
```

/path/to/csv/files: A suggestion is `/scratch/gpfs/$USER/zyg_raw`

When complete, multiple JSON files will be created in the same input folder you specify above; these can then be moved to a folder such as `/scratch/gpfs/$USER/zyg_in` - you'll need this in finetune.yaml as well

# A Test Run

## With capitalization dataset

For full instructions, see the README.md in tests/input_training (NB - if you haven't already, you'll need to the **ttenv-nightly** environment for this test!)

## With twin dataset

Now that we have a yaml/slurm generator, we can leverage that to make the files for our run. Before running, please check:

* Make sure `input_dir_base` is set correctly to your choice in the previous section
* If you have subfolders in `input_dir_base`, make sure to change `input_formatting` to the name of the subfolder you want instead of an empty string (uncommon)
* Make sure `conda_env` is the same one you created previously

```
python generate_slurm_script.py --my_wandb_project my_first_tests --my_wandb_run_name my_first_test --input_dir_base /scratch/gpfs/$USER/zyg_in/ --input_formatting '' --conda_env ttenv
```

Then run `sbatch finetune_filled.slurm` and watch the magic happen!


