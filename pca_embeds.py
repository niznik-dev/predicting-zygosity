# %%
import os   
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


from utils.llm_utils import load_tensor_with_ids


def run_pca_on_embeddings(embeddings, n_components=2, random_state=42, **kwargs):
    """
    Performs PCA on the provided embeddings and optionally saves a 2D scatter plot.

    Parameters:
    -----------
    embeddings : np.ndarray
        The full embedding matrix (n_samples x embedding_dim).
    
    n_components : int, default=2
        Number of principal components to retain.
    
    random_state : int, default=42
        Random seed for reproducibility.

    **kwargs : dict
        Additional keyword arguments for sklearn.decomposition.PCA.

    Returns:
    --------
    pca_model : sklearn.decomposition.PCA
        The fitted PCA model (with principal components).
    
    reduced_embeddings : np.ndarray
        The low-dimensional embedding projections. Shape: (n_samples, n_components)
    """

    # Fit PCA
    pca_model = PCA(n_components=n_components, random_state=random_state, **kwargs)
    reduced_embeddings = pca_model.fit_transform(embeddings)

    return pca_model, reduced_embeddings



def plot_pca_scatter(reduced_embeddings, labels=None, title="PCA Scatter Plot", save_path=None):
    """
    Plots a 2D scatter plot of PCA-reduced embeddings.

    Parameters:
    -----------
    reduced_embeddings : np.ndarray
        The low-dimensional embedding projections. Shape: (n_samples, 2)
    
    labels : list or np.ndarray, optional
        Labels for each point in the scatter plot. If provided, points will be colored by label.
    
    title : str, default="PCA Scatter Plot"
        Title of the plot.
    
    save_path : str, optional
        If provided, saves the plot to this path.
    """
    
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        # # labels are pd Series, check if entried in the label are string or numeric
        # if np.issubdtype(labels.dtype, np.number):
        #     # we use continous color map
        #     scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', alpha=0.7)
        #     plt.colorbar(scatter

        if type(labels[0]) == str:
            unique_labels = np.unique(labels)
            for label in unique_labels:
                idx = np.where(labels == label)
                plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1], label=label)
        else:
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
        plt.legend()
    else:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.close()


# %% 
# ----------- Common Directories -----------
RUN_NAME="1B-100k-20epoch" # name of folder with checkpoints
BASE_DIR="/home/drigobon/scratch/"
DATA_PATH=f"{BASE_DIR}/BoL-in/book_of_life_paraphrases.1K.json"
RAW_DATA_PATH=f"{BASE_DIR}/BoL-in/book_of_life_paraphrases.1K.csv"
CHECKPOINT_BASE_DIR=f"{BASE_DIR}/BoL-out/{RUN_NAME}/" # Base directory for checkpoints (unused currently)
LOAD_PATH_BASE = f"{BASE_DIR}/preprompt_test/hidden_states/"
SAVE_PATH_BASE = f"{BASE_DIR}/preprompt_test/pca_plots/"



df = pd.read_csv(RAW_DATA_PATH)
df.head()

label_cluster = []

for i in df.text:
    temp = i.split(sep=' ')

    if temp[0] == 'A':
        label_cluster.append(0)
    elif temp[0] == 'Born':
        label_cluster.append(1)
    elif temp[0] == 'This':
        label_cluster.append(2)
    elif temp[2] == 'lives':
        label_cluster.append(3)
    elif temp[1] == 'and':
        label_cluster.append(4)
    else:
        raise ValueError(f"Unexpected text format: {i}")

df['cluster_label'] = label_cluster


# %%


POOL_TYPES = ['mean_non_padding', 'last_non_padding']

# ----------- Embedding & Pooling Params -----------
PREPROMPTS={
    'default': '',
    'DR': "You are a highly educated medical specialist. The following is a short description of a person, who is currently living in the Netherlands. Your task is to determine if this individual exhibits any risk of cardiovascular disease. Provide a response of 1 if they do, and 0 if they do not. Respond with only a single number, either 1 or 0. Do not provide any additional information or explanation.",
    'KS': "You are an AI assistant that is a social science specialist. Summarize the main information about this person for registry-level retrieval task. Be concise, focus on factual information.",
    'MS': "You are a very smart language model. I'd like you to summarize the information in this story about a person. Please focus on what is important about the essence of the person.  Now what is one word you'd use to describe the essence of this person.",
    'AM': "Pretend you've been fine tuned for a task. Try to figure out what is the right target grouping of these people.",
    'MN': "Hey buddy, do your best!",
}

color_var = 'gender'

max_layers = 17



for POOL_TYPE in POOL_TYPES:
    print(f"\nPooling type: {POOL_TYPE}")

    for preprompt_name, preprompt in PREPROMPTS.items():
        
        print(f"\nUsing preprompt: {preprompt_name} - '{preprompt}'")

        LOAD_PATH = f"{LOAD_PATH_BASE}/{preprompt_name}/embeds_pooled_{POOL_TYPE}.h5"

        # Load the embeddings
        embeddings, ids, _ = load_tensor_with_ids(LOAD_PATH)


        for layer in range(max_layers):

            SAVE_PATH = f"{SAVE_PATH_BASE}/layer_{layer}/{POOL_TYPE}/"

            # Create the directory if it doesn't exist
            os.makedirs(SAVE_PATH, exist_ok=True)


            # Run PCA and plot the results
            pca_model, reduced_embeddings = run_pca_on_embeddings(embeddings[:,layer,:], n_components=2)

            # Plot the PCA results
            plot_pca_scatter(reduced_embeddings, labels = df[color_var], 
                            title=f"PCA Scatter Plot - {preprompt_name} ({POOL_TYPE})",
                            save_path=f"{SAVE_PATH}/{preprompt_name}_pca_scatter_color_by_{color_var}.png")


# %%