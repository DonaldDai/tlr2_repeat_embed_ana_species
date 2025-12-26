import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.decomposition import PCA

def get_raw_embeddings(data, uniprot_id, start=0, end=None):
    """
    Extract raw embeddings for a specific UniProt ID and range.
    """
    p_data = data[data['uniprot_id'] == uniprot_id]
    if p_data.empty:
        return None
    
    p_data = p_data.sort_values(by='position', ascending=True)
    
    # Handle range
    total_len = len(p_data)
    actual_end = end if end is not None else total_len
    actual_start = start
    
    # Slice
    subset = p_data.iloc[actual_start:actual_end]
    if subset.empty:
        return None
        
    embeddings = np.array(subset['embedding'].tolist())
    return embeddings

def smooth_and_normalize(embeddings, window_size=9):
    """
    Apply sliding window smoothing and L2 normalization.
    """
    L, D = embeddings.shape
    
    # Smoothing
    kernel = np.ones(window_size) / window_size
    smoothed_emb = np.array([
        np.convolve(embeddings[:, d], kernel, mode='same') 
        for d in range(D)
    ]).T
    
    # L2 Normalization
    norm = np.linalg.norm(smoothed_emb, axis=1, keepdims=True)
    normalized_emb = smoothed_emb / (norm + 1e-10)
    
    return normalized_emb

def compute_cross_similarity(emb1, emb2):
    """
    Compute cross-similarity matrix between two sets of embeddings.
    """
    return np.dot(emb1, emb2.T)

if __name__ == "__main__":
    # ==========================================
    # Configuration
    # ==========================================
    
    # Define targets: (UniProt ID, Label, (Start, End))
    # Set End to None for full length
    TARGETS = [
        {'id': 'O60603', 'name': 'Human[O60603]', 'range': (20, 580)},
        {'id': 'S4S1Q8', 'name': 'Lamprey[S4S1Q8]', 'range': (20, 590)},
        {'id': 'UPI00358FFA15', 'name': 'Hagfish[UPI00358FFA15]', 'range': (0, None)},
        {'id': 'T1EUA2', 'name': 'Leech[T1EUA2]', 'range': (20, 640)},
        {'id': 'F6SGF2', 'name': 'Ciona[F6SGF2]', 'range': (10, 730)},
    ]
    
    # Analysis Parameters
    LAYERS = range(1, 25) # All layers
    WINDOW_SIZE = 3
    USE_PCA = True
    PCA_VARIANCE = 0.95
    
    # Directories
    DATA_DIR = './layer_embeds_species'
    SAVE_DIR = './sim_matrices_species'
    FIG_DIR = './sim_fig_species'
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # Store processed data for all layers: processed_data[layer][uid] = embedding
    processed_data = {}

    # 1. Load and Process Data for ALL Layers
    print("Step 1: Loading and Processing Data for all layers...")
    for layer in LAYERS:
        print(f"  Processing Layer {layer}...")
        file_path = f"{DATA_DIR}/TLR2_prot_t5_layer_{layer}.parquet"
        
        try:
            data = pd.read_parquet(file_path)
        except FileNotFoundError:
            print(f"    File not found: {file_path}")
            continue

        # Collect Raw Embeddings
        target_embeddings = {}
        all_embeddings_list = []
        
        for target in TARGETS:
            uid = target['id']
            rng = target['range']
            raw_emb = get_raw_embeddings(data, uid, rng[0], rng[1])
            
            if raw_emb is not None:
                target_embeddings[uid] = raw_emb
                all_embeddings_list.append(raw_emb)
            else:
                print(f"    Warning: No data for {uid} in layer {layer}")

        if not target_embeddings:
            continue

        # PCA (Optional) - Fit on ALL data for this layer
        if USE_PCA:
            combined_emb = np.vstack(all_embeddings_list)
            pca = PCA(n_components=PCA_VARIANCE)
            pca.fit(combined_emb)
            # Transform each
            for uid in target_embeddings:
                target_embeddings[uid] = pca.transform(target_embeddings[uid])

        # Smoothing and Normalization
        processed_data[layer] = {}
        for uid, emb in target_embeddings.items():
            processed_data[layer][uid] = smooth_and_normalize(emb, window_size=WINDOW_SIZE)

    # 2. Generate Plots for Each Pair
    print("Step 2: Generating Plots for each pair...")
    n_targets = len(TARGETS)
    
    for i in range(n_targets):
        for j in range(n_targets): # Iterate all pairs (including self and symmetric)
            target_i = TARGETS[i]
            target_j = TARGETS[j]
            uid_i = target_i['id']
            uid_j = target_j['id']
            name_i = target_i['name']
            name_j = target_j['name']
            
            print(f"  Plotting {name_i} vs {name_j}...")
            
            # Prepare Figure: 4 rows x 6 columns = 24 plots
            cols = 6
            rows = 4
            fig, axes = plt.subplots(rows, cols, figsize=(24, 16))
            fig.suptitle(f"Comparison: {name_i} vs {name_j} (Window={WINDOW_SIZE})", fontsize=24)
            axes = axes.flatten()
            
            has_data = False
            for idx, layer in enumerate(LAYERS):
                ax = axes[idx]
                
                if layer in processed_data and uid_i in processed_data[layer] and uid_j in processed_data[layer]:
                    emb_i = processed_data[layer][uid_i]
                    emb_j = processed_data[layer][uid_j]
                    
                    # Compute Similarity
                    sim_matrix = compute_cross_similarity(emb_i, emb_j)
                    
                    sim_parquet_path = f"{SAVE_DIR}/cross_sim_matrix_{name_i}_vs_{name_j}_win_{WINDOW_SIZE}_layer_{layer}.parquet"
                    pd.DataFrame(sim_matrix).to_parquet(sim_parquet_path)
                    print(f"Saved similarity matrix to {sim_parquet_path}")
                    
                    # Plot
                    im = ax.imshow(sim_matrix, cmap='viridis', vmin=0, vmax=1, origin='upper')
                    ax.set_title(f"Layer {layer}")
                    
                    # Only add labels to outer plots to save space
                    if idx % cols == 0:
                        ax.set_ylabel(f"{name_i}\n({sim_matrix.shape[0]} aa)", fontsize=8)
                    if idx >= (rows - 1) * cols:
                        ax.set_xlabel(f"{name_j}\n({sim_matrix.shape[1]} aa)", fontsize=8)
                        
                    has_data = True
                else:
                    ax.text(0.5, 0.5, "Missing Data", ha='center', va='center')
                    ax.set_title(f"Layer {layer}")
                    ax.axis('off')

            if has_data:
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                output_file = f'{FIG_DIR}/Cross_Sim_Layers_{name_i}_vs_{name_j}_win_{WINDOW_SIZE}.jpg'
                # Sanitize filename
                output_file = output_file.replace(" ", "_").replace("(", "").replace(")", "")
                plt.savefig(output_file, dpi=100)
                print(f"    Saved to {output_file}")
            else:
                print(f"    Skipping {name_i} vs {name_j} (No Data)")
            
            plt.close(fig)
