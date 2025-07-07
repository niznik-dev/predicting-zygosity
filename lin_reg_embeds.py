
import os   
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import arviz as az

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score

from utils.llm_utils import load_tensor_with_ids


# ----------- Inputs -----------
BASE_DIR="/home/drigobon/scratch/"
RUN_NAME="1B-100k-20epoch" # name of folder with checkpoints

DATA_PATH=f"{BASE_DIR}/BoL-in/book_of_life_biased.no_label_in_text.100K.json"
RAW_DATA_PATH=f"{BASE_DIR}/BoL-in/book_of_life_biased.no_label_in_text.100K.csv"
CHECKPOINT_BASE_DIR=f"{BASE_DIR}/BoL-out/{RUN_NAME}/"
LOAD_PATH_BASE = f"{CHECKPOINT_BASE_DIR}/hidden_states/"
SAVE_PATH_BASE = f"{CHECKPOINT_BASE_DIR}/figs-r2-vs-layer/"


MAX_EPOCHS = 5
MAX_LAYERS = 17
POOL_TYPES = ['mean_non_padding', 'last_non_padding']

NUM_OBS = 10000


# Define fitting model
model = LogisticRegression(max_iter=1000)
# model = LinearRegression()

# ----------- End Inputs -----------





print(f"Loading data from {DATA_PATH}")
df = pd.read_csv(RAW_DATA_PATH, usecols = ['cardio_history'])

if NUM_OBS is None:
    NUM_OBS = len(df)

Ys = df.cardio_history.iloc[:NUM_OBS].values
print(f"Loaded {len(Ys)} labels")



# Over Pooling Types
for POOL_TYPE in POOL_TYPES:
    print(f"\nPooling type: {POOL_TYPE}")

    # Make subplots
    f, ax = plt.subplots(MAX_LAYERS, 1+MAX_EPOCHS, figsize=(5*(1+MAX_EPOCHS), 1*MAX_LAYERS))
    f.tight_layout(pad=3.0)  # Add padding between subplots for clarity


    # Base model first
    LOAD_PATH = f"{LOAD_PATH_BASE}/base_model/embeds_pooled_{POOL_TYPE}.h5"
    SAVE_PATH = f"{SAVE_PATH_BASE}/base_model/"
    
    # Create the directory if it doesn't exist
    os.makedirs(SAVE_PATH, exist_ok=True)

    # Load the embeddings
    embeddings, ids, _ = load_tensor_with_ids(LOAD_PATH)

    # Subset the embeddings and labels to NUM_OBS
    embeddings = embeddings[:NUM_OBS, :, :]  # Shape: (NUM_OBS, MAX_LAYERS, EMBEDDING_DIM)

    r2s = []

    for layer in range(MAX_LAYERS):

        Xs = embeddings[:, layer, :]

        model.fit(Xs, Ys)

        # Get predicted
        Ys_pred = model.predict_proba(Xs)[:, 1] 

        # Get r2 value
        r2s.append(r2_score(Ys, Ys_pred))

        # Add to subplot
        az.plot_separation(y = Ys, y_hat = Ys_pred, ax=ax[layer, 0])
        ax[layer, 0].set_title(f"Layer {1+layer} - Base Model - Pooling: {POOL_TYPE}")
        
    # Plot r2 vs layer
    plt.figure(figsize=(10, 6))
    plt.plot(range(MAX_LAYERS), r2s, marker='o', linestyle='-')
    plt.title(f"R2 vs Layer for Base Model - Pooling: {POOL_TYPE}")
    plt.xlabel("Layer")
    plt.ylabel("R2 Score")
    plt.savefig(f"{SAVE_PATH}/{POOL_TYPE}_r2_vs_layer.png")
    plt.close()



    # Loop through finetuning epochs
    for i in range(MAX_EPOCHS):
        print(f"\nProcessing epoch {i}")        

        LOAD_PATH = f"{LOAD_PATH_BASE}/epoch_{i}/embeds_pooled_{POOL_TYPE}.h5"
        SAVE_PATH = f"{SAVE_PATH_BASE}/epoch_{i}/"
        
        # Create the directory if it doesn't exist
        os.makedirs(SAVE_PATH, exist_ok=True)

        # Load the embeddings
        embeddings, ids, _ = load_tensor_with_ids(LOAD_PATH)

        # Subset the embeddings and labels to NUM_OBS
        embeddings = embeddings[:NUM_OBS, :, :]  # Shape: (NUM_OBS, MAX_LAYERS, EMBEDDING_DIM)

        r2s = []

        for layer in range(MAX_LAYERS):

            Xs = embeddings[:, layer, :]

            model.fit(Xs, Ys)

            # Get predicted
            Ys_pred = model.predict_proba(Xs)[:, 1] 
            
            # Get r2 value
            r2s.append(r2_score(Ys, Ys_pred))

            # Add to subplot
            az.plot_separation(y = Ys, y_hat = Ys_pred, ax=ax[layer, i+1])
            ax[layer, i+1].set_title(f"Layer {1+layer} - Epoch {i} - Pooling: {POOL_TYPE}")
            
        # Plot r2 vs layer
        plt.figure(figsize=(10, 6))
        plt.plot(range(MAX_LAYERS), r2s, marker='o', linestyle='-')
        plt.title(f"R2 vs Layer for Epoch {i} - Pooling: {POOL_TYPE}")
        plt.xlabel("Layer")
        plt.ylabel("R2 Score")
        plt.savefig(f"{SAVE_PATH}/{POOL_TYPE}_r2_vs_layer.png")
        plt.close()


    # Save subplots
    f.savefig(f"{SAVE_PATH_BASE}/{POOL_TYPE}_separation_plots.png")
    plt.close(f)



