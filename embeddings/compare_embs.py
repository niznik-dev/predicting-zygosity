import numpy as np
import pandas as pd
import argparse
import json
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def load_prompts_and_targets(eval_file, num_obs = None):
    """
    Loads prompts and target outputs from a JSON evaluation file.
    Args:
        eval_file (str): Path to the JSON file containing evaluation data. The file should contain a list of dictionaries,
            each with at least the keys "input" (prompt) and "output" (target).
        num_obs (int or None): Number of observations to load. If None or negative, defaults to the smaller of 20 or the
            total number of examples in the file.
    Returns:
        tuple:
            - prompts (list of str): List of prompt strings extracted from the evaluation data.
            - targets (list of str): List of target output strings (stripped of leading/trailing whitespace) corresponding to the prompts.
    Example:
        prompts, targets = load_prompts_and_targets("eval_data.json", 10)
    """
    # ─── Load evaluation data ─────────────────────────────────────────────────────
    with open(eval_file, "r") as f:
        eval_data = json.load(f)

    # ─── Extract prompts & targets for desired number of observations ─────────────
    if num_obs is None or num_obs < 0:
        num_obs = len(eval_data)  # default to number of observations in the eval data

    prompts = [ex["input"] for ex in eval_data][0:num_obs]
    targets = [ex["output"].strip() for ex in eval_data][0:num_obs]

    return prompts, targets

def load_embeddings(filepath, num_obs = None):
    """
    Loads embeddings data.

    Parameters:
    -----------
    filepath : str
        Filename for the .json file with embedding data.
    num_obs : int or None
        Number of observations to load. If None, defaults to all observations in the file.

    Returns:
    --------
    embeddings : np.ndarray
        The embeddings. Shape: (n_obs, embedding_dim)
    """

    with open(filepath, "r") as f:
        data = json.load(f)
    embeddings_full = np.array(data)[:num_obs, :] if num_obs is not None else np.array(data)

    return embeddings_full


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

def run_tsne_on_embeddings(embeddings, n_components=2, random_state=42, **kwargs):
    """
    Performs t-SNE on the provided embeddings and returns the low-dimensional projections.

    Parameters:
    -----------
    embeddings : np.ndarray
        The full embedding matrix (n_samples x embedding_dim).

    n_components : int, default=2
        Number of dimensions to reduce to.

    random_state : int, default=42
        Random seed for reproducibility.

    **kwargs : dict
        Additional keyword arguments for sklearn.manifold.TSNE.

    Returns:
    --------
    reduced_embeddings : np.ndarray
        The low-dimensional embedding projections. Shape: (n_samples, n_components)
    """

    tsne = TSNE(n_components=n_components, random_state=random_state, **kwargs)
    reduced_embeddings = tsne.fit_transform(embeddings)

    return reduced_embeddings

def run_umap_on_embeddings(embeddings, n_components=2, random_state=42, **kwargs):
    """
    Performs UMAP on the provided embeddings and returns the low-dimensional projections.

    Parameters:
    -----------
    embeddings : np.ndarray
        The full embedding matrix (n_samples x embedding_dim).

    n_components : int, default=2
        Number of dimensions to reduce to.

    random_state : int, default=42
        Random seed for reproducibility.

    **kwargs : dict
        Additional keyword arguments for umap.UMAP.

    Returns:
    --------
    umap_model : umap.UMAP
        The fitted UMAP model.

    reduced_embeddings : np.ndarray
        The low-dimensional embedding projections. Shape: (n_samples, n_components)
    """

    umap_model = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
    reduced_embeddings = umap_model.fit_transform(embeddings)

    return umap_model, reduced_embeddings

# For sorting the embeddings files outputted by ttune (default to epoch_* without a uniform number of digits)
def sort_key(item):
    name, _ = item
    if name == 'base':
        return (0, 0)
    elif name.startswith('epoch_'):
        try:
            epoch_num = int(name.split('_')[1])
        except Exception:
            epoch_num = float('inf')
        return (1, epoch_num)
    else:
        return (2, name)

# ─── Argument Parsing ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Compare and visualize embedding projections over finetuning checkpoints.")

parser.add_argument(
    '--emb_type', 
    type=str, 
    choices=['PCA', 'TSNE', 'UMAP'], 
    required=True,
    help='Method of dimensionality reduction.'
)

parser.add_argument(
    '--fit_to_base', 
    type=int, 
    choices=[0, 1], 
    default=0,
    help='Fit the PCA/UMAP model to the base embeddings (only for PCA/UMAP).'
)

parser.add_argument(
    '--n_components', 
    type=int, 
    default=3,
    help='Number of principal components to extract (2D or 3D plots).'
)

parser.add_argument(
    '--embeddings_dir', 
    type=str, 
    required=True,                
    help='Directory containing embedding files.'
)

parser.add_argument(
    '--pattern', 
    type=str, 
    default="epoch_*.json",
    help='Pattern for model checkpoint embedding files.'
)

parser.add_argument(
    '--fig_dir_base', 
    type=str, 
    required=True,
    help='Base directory for saving figures.'
)

parser.add_argument(
    '--eval_file', 
    type=str, 
    required=True,                
    help='Path to the evaluation file with prompts and targets.'
)

parser.add_argument(
    '--num_obs', 
    type=int, 
    help='Number of observations to use from the embeddings file.'
)

parser.add_argument(
    '--save_plots_2d', 
    type=int, 
    choices=[0, 1], 
    default=0,
    help='Save the 2D plots.'
)

parser.add_argument(
    '--save_plots_3d', 
    type=int, 
    choices=[0, 1], 
    default=0,
    help='Save the 3D plots.'
)

parser.add_argument(
    '--save_plots_grid', 
    type=int, 
    choices=[0, 1], 
    default=0,
    help='Save the 2D plots in a grid.'
)

parser.add_argument(
    '--k_epoch', 
    type=int, 
    default=5,
    help='Number of epochs per subplot row in the grid.'
)

parser.add_argument(
    '--save_plots_cos_sim', 
    type=int, 
    choices=[0, 1], 
    default=0,
    help='Save the cosine similarity plots (only for PCA).'
)

parser.add_argument(
    '--max_pc_to_compare', 
    type=int, 
    default=1,
    help='Maximum principal component to compare for cosine similarity (cannot be greater than n_components).'
)

args = parser.parse_args()


# ─── Print Arguments ───────────────────────────────────────────────────────
print("Arguments:")
for arg, value in vars(args).items():
    print(f"{arg}: {value}")



# Assign variables from args
eval_file = args.eval_file
embeddings_dir = args.embeddings_dir
pattern = args.pattern
num_obs = args.num_obs
emb_type = args.emb_type
fig_dir_base = args.fig_dir_base
fit_to_base = bool(args.fit_to_base)
n_components = args.n_components
save_plots_2d = bool(args.save_plots_2d)
save_plots_3d = bool(args.save_plots_3d)
save_plots_grid = bool(args.save_plots_grid)
k_epoch = args.k_epoch
save_plots_cos_sim = bool(args.save_plots_cos_sim)
max_pc_to_compare = args.max_pc_to_compare




# Check UMAP if needed
if emb_type == 'UMAP':
    try:
        import umap
    except ImportError:
        raise ImportError("UMAP is not installed.")




# ─── Figure Sub-Directories  ───────────────────────────────────────────────────────
fig_dir_2d = fig_dir_base + "2D/"
fig_dir_3d = fig_dir_base + "3D/"

Path(fig_dir_base).mkdir(parents=True, exist_ok=True)
if save_plots_2d:
    Path(fig_dir_2d).mkdir(parents=True, exist_ok=True)
if save_plots_3d:
    Path(fig_dir_3d).mkdir(parents=True, exist_ok=True)



# ─── Define axes labels and titles for emb_type ───────────────────────────────────────────────────────
if emb_type == 'PCA':
    if fit_to_base:
        x_label = "Base Model PC1"
        y_label = "Base Model PC2"
        z_label = "Base Model PC3"
        title_2d = "PCA 2D Projection (fit to base)"
        title_3d = "PCA 3D Projection (fit to base)"
    else:
        x_label = "PC1"
        y_label = "PC2"
        z_label = "PC3"
        title_2d = "PCA 2D Projection"
        title_3d = "PCA 3D Projection"
elif emb_type == 'UMAP':
    x_label = "UMAP 1"
    y_label = "UMAP 2"
    z_label = "UMAP 3"
    title_2d = "UMAP 2D Projection"
    title_3d = "UMAP 3D Projection"
elif emb_type == 'TSNE':
    x_label = "t-SNE 1"
    y_label = "t-SNE 2"
    z_label = "t-SNE 3"
    title_2d = "t-SNE 2D Projection"
    title_3d = "t-SNE 3D Projection"
else:
    raise ValueError(f"Unknown embedding type: {emb_type}.")

# ─── Load evaluation data ─────────────────────────────────────────────────────
(prompts, targets) = load_prompts_and_targets(eval_file, num_obs)
targets = np.array(targets).astype(int)  # convert to numpy array for easier manipulation later


# ─── Find all embedding files from model checkpoints ───────────────────────────────────────────────────────
embeddings_files = {'base': embeddings_dir+'base.json'}
print(f"Looking for embeddings files in {embeddings_dir} with pattern {pattern}...")
for item in Path(embeddings_dir).iterdir():
    if item.match(pattern):

        name = str(item).split('/')[-1].split('.')[0]

        if len(name) > 0:
            embeddings_files[name] = str(item)
print(f"Found {len(embeddings_files.items())} embeddings files:")
print(embeddings_files)


# ─── Extract PC's for different embeddings ───────────────────────────────────────────────────────
reduced_embs = dict()

if (emb_type=='PCA') and save_plots_cos_sim: # only if PCA is used, and cosine similarity plots are requested
    PCs = dict()

if fit_to_base:
    mod_base = None
    reduced_embs_base = dict()  # to store base embeddings if needed later for plotting

# Note: base embeddings must be loaded first, so that they can be used for fitting PCA/UMAP models if fit_to_base is True
for name, file in sorted(embeddings_files.items(), key=sort_key):
    print(f"Processing {name} embeddings from {file}...")

    # ─── Load embeddings ───────────────────────────────────────────────────────
    embeddings = load_embeddings(file, num_obs)
    print(f"Loaded {embeddings.shape[0]} embeddings with shape {embeddings.shape[1]} from {file}.")
    
    if emb_type == 'TSNE':
        # ─── Run t-SNE on embeddings ───────────────────────────────────────────────────────
        reduced_emb_curr = run_tsne_on_embeddings(
                                        embeddings, 
                                        n_components = n_components
                                    )
    elif emb_type == 'PCA':
        # ─── Run PCA on embeddings ───────────────────────────────────────────────────────
        pca_curr, reduced_emb_curr = run_pca_on_embeddings(
                                        embeddings, 
                                        n_components = n_components
                                    )
        # Save base PCA model if needed later for plotting
        if name == 'base' and fit_to_base:
            mod_base = pca_curr
            reduced_embs_base[name] = reduced_emb_curr
        # For other models fit to base PC space if fit_to_base is True
        elif fit_to_base:
            reduced_embs_base[name] = mod_base.transform(embeddings)
        
        if save_plots_cos_sim:
            PCs[name] = pca_curr.components_
            
    elif emb_type == 'UMAP':
        # ─── Run UMAP on embeddings ───────────────────────────────────────────────────────
        umap_curr, reduced_emb_curr = run_umap_on_embeddings(
                                        embeddings, 
                                        n_components = n_components
                                    )
        # Save base UMAP model if needed later for plotting
        if name == 'base' and fit_to_base:
            mod_base = umap_curr
            reduced_embs_base[name] = reduced_emb_curr
        # For other models fit to base PC space if fit_to_base is True
        elif fit_to_base:
            reduced_embs_base[name] = mod_base.transform(embeddings)
        
    reduced_embs[name] = reduced_emb_curr




# ─── Individual 2D and 3D Plots ───────────────────────────────────────────────────────
for name, file in sorted(embeddings_files.items(), key=sort_key):

    if save_plots_2d and n_components >= 2:
        # ─── Projection PC's colored by targets (2D) ───────────────────────────────────────────────────────
        fig, ax = plt.subplots(1,1,figsize=(10, 8))
        
        if fit_to_base and (emb_type == 'PCA' or emb_type == 'UMAP'):
            ax.scatter(
                reduced_embs_base[name][:, 0],
                reduced_embs_base[name][:, 1],
                c=targets,
                cmap='viridis',
                alpha=0.7
            )
        else:
            ax.scatter(
                reduced_embs[name][:, 0],
                reduced_embs[name][:, 1],
                c=targets,
                cmap='viridis',
                alpha=0.7
            )

        ax.set_title(f"{title_2d} of {name} embeddings colored by targets")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.tight_layout()
        plt.savefig(f"{fig_dir_2d}{name}_PCs_colored_by_targets_2D.png")
        plt.close()

    if save_plots_3d and n_components >= 3:
        # ─── Projection PC's colored by targets (3D) ───────────────────────────────────────────────────────
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if fit_to_base and (emb_type == 'PCA' or emb_type == 'UMAP'):
            sc = ax.scatter(
                reduced_embs_base[name][:, 0],
                reduced_embs_base[name][:, 1],
                reduced_embs_base[name][:, 2],
                c=targets,
                cmap='viridis',
                alpha=0.7
            )
        else:
            sc = ax.scatter(
                reduced_embs[name][:, 0],
                reduced_embs[name][:, 1],
                reduced_embs[name][:, 2],
                c=targets,
                cmap='viridis',
                alpha=0.7
            )



        
        ax.set_title(f"{title_3d} of {name} embeddings colored by targets")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        fig.colorbar(sc, ax=ax, label='Target')
        plt.tight_layout()
        plt.savefig(f"{fig_dir_3d}{name}_PCs_colored_by_targets_3D.png")
        plt.close()




# ─── Plot all models in a grid ───────────────────────────────────────────────────────

# Collect names in sorted order for consistent plotting
sorted_names = [name for name, _ in sorted(embeddings_files.items(), key=sort_key)]

if save_plots_grid and n_components >= 2:
    n_models = len(sorted_names)
    ncols = k_epoch
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for idx, name in enumerate(sorted_names):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.scatter(
            reduced_embs[name][:, 0],
            reduced_embs[name][:, 1],
            c=targets,
            cmap='viridis',
            alpha=0.7
        )
        ax.set_title(f"{name}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    # Hide unused subplots
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis('off')
    plt.tight_layout()
    plt.savefig(f"{fig_dir_base}all_models_{title_2d}_colored_by_targets_grid.png")
    plt.close()






    # ─── Compare cosine similarity of PC's over time (only for PCA emb_type) ───────────────────────────────────────────────────────
    if emb_type == 'PCA':
        for pc_to_compare in range(1, max_pc_to_compare + 1):
            v1 = PCs['base'][pc_to_compare - 1, :]  # Baseline PC

            cos_sim = {}
            epochs = []
            sim_values = []

            # Sort keys so 'base' is first, then epochs numerically
            for name, file in sorted(embeddings_files.items(), key=sort_key):
                v2 = PCs[name][pc_to_compare - 1, :]
                # Handle possible zero division
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                if norm_v1 == 0 or norm_v2 == 0:
                    cos_sim_curr = np.nan
                else:
                    cos_sim_curr = np.dot(v1, v2) / (norm_v1 * norm_v2)
                cos_sim[name] = cos_sim_curr

                # For x-axis labeling
                if name == 'base':
                    epochs.append(0)
                elif name.startswith('epoch_'):
                    try:
                        epochs.append(int(name.split('_')[1]))
                    except Exception:
                        epochs.append(np.nan)
                else:
                    epochs.append(np.nan)
                sim_values.append(cos_sim_curr)

            # Sort by epochs for plotting
            if save_plots_cos_sim:
                epochs, sim_values = zip(*sorted(zip(epochs, sim_values)))
                plt.figure(figsize=(8, 6))
                plt.plot(epochs, sim_values, marker='o')
                plt.title(f"Cosine similarity between PC{pc_to_compare} of base and fine-tuned model")
                plt.xlabel("Number of Training Epochs")
                plt.ylabel("Cosine Similarity")
                plt.xticks(epochs)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{fig_dir_base}CosSimOverEpochs_PC{pc_to_compare}.png")
                plt.close()
