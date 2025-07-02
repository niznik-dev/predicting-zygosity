import math
import os
import re
import yaml

from itertools import product

# First, let's do some calculations based on the parameters that will
# change the amount of memory needed; we will make an estimate and then
# vary the other params in a slurm file adapted to that combination
def estimate_memory(scaling_factor, memory_multipliers):
    """
    Estimate the memory (in GB) required based on the scaling factor and memory multipliers.

    Args:
        scaling_factor (float): The scaling factor for the memory.
        memory_multipliers (list): A list of multipliers for different memory components.
    Returns:
        float: Estimated memory in GB.
    """
    return math.ceil(scaling_factor * math.prod(memory_multipliers))


def main():
    with open('perm_params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    SCALING_FACTOR = params["scaling_factor"]
    MEMORY_MULTIPLIERS = params["memory_multipliers"]

    OUTSIDE_VARIABLES = list(params.get("outside_variables", {}).keys())
    OUTSIDE_VARIABLES_NOMULT = [v for v in OUTSIDE_VARIABLES if v not in MEMORY_MULTIPLIERS]
    TORCHTUNE_VARIABLES = list(params.get("torchtune_variables", {}).keys())
    TORCHTUNE_VARIABLES_NOMULT = [v for v in TORCHTUNE_VARIABLES if v not in MEMORY_MULTIPLIERS]

    # Gather all possible values for each variable in memory_multipliers
    mult_value_lists = []
    for var in MEMORY_MULTIPLIERS:
        value = None
        for section in ["torchtune_variables", "outside_variables"]:
            if section in params and var in params[section]:
                value = params[section][var]
                break
        if value is None:
            raise ValueError(f"Variable '{var}' not found in YAML sections.")
        mult_value_lists.append(value)

    with open('finetune_filled.yaml', 'r') as f:
        FINETUNE_PARAMS_BASE = yaml.safe_load(f)

    with open('finetune_filled.slurm', 'r') as f:
        FINETUNE_SLURM_BASE = f.read()
    
    match_job_name = re.search(r'--job-name=(\S+)', FINETUNE_SLURM_BASE)
    SWEEP_NAME = match_job_name.group(1)

    # Now loop over all combinations
    # If mult_value_lists is empty (constant memory case), do the loop once with an empty tuple
    combos = list(product(*mult_value_lists)) if mult_value_lists else [()]
    for combo in combos:
        slurm_out = FINETUNE_SLURM_BASE

        slurm_mem = SCALING_FACTOR if not combo else estimate_memory(SCALING_FACTOR, combo)
        combo_dict = dict(zip(MEMORY_MULTIPLIERS, combo)) if MEMORY_MULTIPLIERS else {}

        # Replace the memory value in slurm_out using a wildcard regex to match any number before 'G'
        slurm_out = re.sub(r'(--mem=)[0-9.]+G', rf'\g<1>{slurm_mem}G', slurm_out)
        
        # Create the directory for this particular memory_multiplier combination
        output_dir = f"{SWEEP_NAME}_{slurm_mem}G"
        os.makedirs(output_dir)

        # Gather all possible values for non-memory variables
        outside_nomult_values = [params['outside_variables'][var] for var in OUTSIDE_VARIABLES_NOMULT]
        torchtune_nomult_values = [params['torchtune_variables'][var] for var in TORCHTUNE_VARIABLES_NOMULT]

        combo_index = 0

        # Loop over all combinations of non-memory variables
        for outside_combo in product(*outside_nomult_values):
            for torchtune_combo in product(*torchtune_nomult_values):
                yaml_out = FINETUNE_PARAMS_BASE.copy()
                combos_full = combo_dict.copy()
                combos_full.update(dict(zip(OUTSIDE_VARIABLES_NOMULT, outside_combo)))
                combos_full.update(dict(zip(TORCHTUNE_VARIABLES_NOMULT, torchtune_combo)))
                # print(f"Memory: {slurm_mem:.2f} GB, Params: {combos_full}")

                for key, value in combos_full.items():
                    if key in OUTSIDE_VARIABLES:
                        yaml_out[f'hsweeper_{key}'] = value
                    else:
                        if '/' in key:
                            # For lora, we only set rank so we can derive alpha
                            if key == 'model/lora_rank':
                                yaml_out['model']['lora_rank'] = value
                                yaml_out['model']['lora_alpha'] = value * 2
                            else:
                                main_key, sub_key = key.split('/', 1)
                                yaml_out[main_key][sub_key] = value
                        else:
                            yaml_out[key] = value

                # TODO - change path to recipe if custom

                yaml_out['my_wandb_run_name'] = f"{SWEEP_NAME}_{slurm_mem}G_{combo_index}"
                yaml_out['output_dir'] = yaml_out['output_dir'][:-1] + f"_{slurm_mem}G_{combo_index}/"

                # Write YAML file
                with open(os.path.join(output_dir, f'{combo_index}.yaml'), 'w') as f:
                    yaml.dump(yaml_out, f)
                combo_index += 1

                if combo_index % 100 == 0:
                    print(f"Generated {combo_index} combinations for {slurm_mem} GB memory.")

        slurm_out = slurm_out.replace("<ARRAY_MAX>", str(combo_index - 1))
        slurm_out = slurm_out.replace("##SBATCH --array", "#SBATCH --array")
        slurm_out = slurm_out.replace(f"{SWEEP_NAME}/", f"{SWEEP_NAME}_{slurm_mem}G_$SLURM_ARRAY_TASK_ID/")
        slurm_out = slurm_out.replace("finetune_filled.yaml", "$SLURM_ARRAY_TASK_ID.yaml")

        with open(os.path.join(output_dir, f'{output_dir}.slurm'), 'w') as f:
            f.write(slurm_out)


if __name__ == "__main__":
    main()