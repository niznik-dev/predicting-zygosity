import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from ast import literal_eval

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    roc_auc_score
)
from sklearn.calibration import calibration_curve

from inspect_ai import (
    Task, 
    task, 
    eval,
)
from inspect_ai.model import GenerateConfig
from inspect_ai.dataset import Sample, FieldSpec, json_dataset

from inspect_ai.scorer import (
    Score, 
    Target,
    scorer, 
    exact, 
    includes,
    accuracy, 
    stderr,
    metric,
    CORRECT,
    INCORRECT,
)
from inspect_ai.solver import (
    generate, 
    chain_of_thought, 
    system_message,
    TaskState
)


## INPUTS
DF_VAL = "/home/niznik/scratch/zyg_in/val-long.json"
#SYSTEM_MESSAGE = f"The following contains textual description of life events for two individuals, generated with statistics Netherlands registry data. These individuals are presented to you in a random order, and your task is to identify if these two individuals are twins. Respond only with '1' if the individuals are twins, and '0' if they are not. Do not provide any additional information or explanation, and do not return more than one integer."
SYSTEM_MESSAGE = ''
MODEL_PATH = "hf//home/niznik/scratch/torchtune_models/Llama-3.2-1B-Instruct"
NUM_OBS = 5
VALID_TARGETS = ["0", "1"]  # Valid outputs for the task
OUTPUT_PREFIX = "/home/niznik/scratch/GitHub/cruijff-kit/inspect/"

config = GenerateConfig(
                max_new_tokens = 2,
                logprobs = True,
                top_logprobs = len(VALID_TARGETS),
        )

inspect_dataset = json_dataset(DF_VAL, 
        FieldSpec(
            input="input",
            target="output",
        )
)
## END INPUTS



@scorer(metrics=[accuracy(), stderr()], higher_is_better=False)
def brier_loss():
    async def score(state: TaskState, target: Target):
        
        top_tokens_and_logprobs = state.output.choices[0].logprobs.content[0].top_logprobs

        prob_dict = {f"raw_prob_{item.token}": float(str(np.round(np.exp(item.logprob), 3))) for item in top_tokens_and_logprobs}
        top_tokens = [item.token for item in top_tokens_and_logprobs]

        # Check that top tokens are the same as valid targets
        if top_tokens != set(VALID_TARGETS):
            print(f"Warning: VALID_TARGETS were not found in the top {len(VALID_TARGETS)} tokens.")
            #raise ValueError(f"VALID_TARGETS were not found in the top {len(VALID_TARGETS)} tokens")

        return Score(
            value=(int(target.text) - prob_dict[f"raw_prob_1"])**2,  # MSE of True binary outcome and probability of 1
            answer=state.output.completion,
            target=target,
            explanation=str(prob_dict),
            correct=None  # Not a binary correct/incorrect, just probability
        )
    return score


@task
def twin_game():
    return Task(
        dataset = inspect_dataset[:NUM_OBS],
        solver = [
            system_message(SYSTEM_MESSAGE),
            generate(config = config)
        ],
        scorer = brier_loss(),
        model = MODEL_PATH,
        config = config,
    )

results = eval(twin_game())

results_df = pd.DataFrame([{'id': item.id, 'target': item.target,
                'answer': item.scores['brier_loss'].answer, 'brier_loss': item.scores['brier_loss'].value} \
                | literal_eval(item.scores['brier_loss'].explanation) for item in results[0].samples])

# Make column for sum of raw probabilities
results_df['sum_raw_probs'] = results_df[[f"raw_prob_{token}" for token in VALID_TARGETS]].sum(axis=1)

# Make columns for normalized probabilities.
for token in VALID_TARGETS:
    results_df[f"norm_prob_{token}"] = results_df[f"raw_prob_{token}"] / results_df['sum_raw_probs']




# Compute classification metrics.
acc = accuracy_score(results_df['target'], results_df['answer'])
precision = precision_score(results_df['target'], results_df['answer'], pos_label="1", average="binary")
recall = recall_score(results_df['target'], results_df['answer'], pos_label="1", average="binary")
f1 = f1_score(results_df['target'], results_df['answer'], pos_label="1", average="binary")
cm = confusion_matrix(results_df['target'], results_df['answer'], labels=["1", "0"])
report = classification_report(results_df['target'], results_df['answer'], labels=["1", "0"])


# Compute AUC score.
auc = roc_auc_score(results_df['target'], results_df['raw_prob_1'])


# Compute calibration curve (Murphy Curve) using sklearn.
fraction_of_positives, mean_predicted_value = calibration_curve(results_df['target'], results_df['raw_prob_1'], n_bins=10)

# Plot and save the calibration (Murphy) curve.
plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibration curve")
plt.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curve (Murphy Curve)")
plt.legend()
murphy_curve_filename = f"{OUTPUT_PREFIX}/{datetime.datetime.now().strftime('%Y-%m-%d')}_murphy_curve.png"
plt.savefig(murphy_curve_filename)
plt.close()


# Save evaluation results to a text file with additional details.
results_filename = f"{OUTPUT_PREFIX}/{datetime.datetime.now().strftime('%Y-%m-%d')}_evalresults.txt"
with open(results_filename, "w") as f:
    f.write(f"Accuracy: {acc * 100:.2f}%\n")
    f.write(f"Precision (Positive=1): {precision:.4f}\n")
    f.write(f"Recall (Positive=1): {recall:.4f}\n")
    f.write(f"F1 Score (Positive=1): {f1:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n")
    f.write("\nClassification Report:\n")
    f.write(report + "\n\n")
    f.write(f"AUC Score: {auc:.4f}\n")
    f.write(f"Murphy Curve saved to: {murphy_curve_filename}\n")
