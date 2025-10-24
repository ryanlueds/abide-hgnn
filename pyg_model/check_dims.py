import torch
import glob
from collections import defaultdict
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

# --- Paths from your dataset.py ---
PATH_HYPERGRAPH = "./data/hypergraphs/"
PATH_ABIDE_LABELS = "./abide/Phenotypic_V1_0b_preprocessed1.csv"
torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])
# --- End paths ---

def check_dataset_health():
    """
    Loops through all .pt files and performs a full health check.
    """
    
    # --- 1. Load Labels ---
    try:
        df = pd.read_csv(PATH_ABIDE_LABELS)
        label_dict = set(df['FILE_ID'])
        print(f"Loaded {len(label_dict)} unique labels from CSV.")
    except FileNotFoundError:
        print(f"Error: Label file not found at {PATH_ABIDE_LABELS}")
        return

    # --- 2. Find all .pt files ---
    all_x_paths = glob.glob(f'{PATH_HYPERGRAPH}*.pt')
    if not all_x_paths:
        print(f"Error: No '.pt' files found in {PATH_HYPERGRAPH}")
        return

    print(f"\nFound {len(all_x_paths)} total files. Starting health check...")

    # --- 3. Setup trackers ---
    feature_dim_counts = defaultdict(int)
    node_counts = []
    
    errors = {
        "missing_label": 0,
        "missing_x": 0,
        "missing_edge_index": 0,
        "index_out_of_bounds": 0,
        "load_error": 0,
    }
    
    valid_files = 0

    # --- 4. Loop and Check ---
    for path in tqdm(all_x_paths):
        filename = os.path.basename(path)
        file_id = filename.removesuffix("_hypergraph.pt")
        
        try:
            # --- Check 1: Label Existence ---
            if file_id not in label_dict:
                errors["missing_label"] += 1
                continue
                
            data = torch.load(path, weights_only=True)
            
            # --- Check 2: Data Existence (x) ---
            if not hasattr(data, 'x') or data.x is None:
                errors["missing_x"] += 1
                continue
                
            # --- Check 3: Data Existence (edge_index) ---
            # We check for 'edge_index' since the loader renames it
            if not hasattr(data, 'edge_index') and not hasattr(data, 'hyperedge_index'):
                errors["missing_edge_index"] += 1
                continue
                
            # --- Check 4: Index Validity ---
            num_nodes = data.x.shape[0]
            edge_idx = data.edge_index if hasattr(data, 'edge_index') else data.hyperedge_index
            
            if edge_idx.numel() > 0: # Check only if it's not empty
                max_node_index = edge_idx[0].max().item()
                if max_node_index >= num_nodes:
                    errors["index_out_of_bounds"] += 1
                    continue
            
            # --- All checks passed, record stats ---
            valid_files += 1
            feature_dim_counts[data.x.shape[1]] += 1
            node_counts.append(num_nodes)
            
        except Exception as e:
            errors["load_error"] += 1
            # print(f"\nError loading {filename}: {e}")

    # --- 5. Print Results ---
    print("\n--- Health Check Complete ---")
    
    print(f"\nTotal Files Scanned: {len(all_x_paths)}")
    print(f"✅ Valid Files: {valid_files}")
    
    print("\n--- Errors Found ---")
    for error_type, count in errors.items():
        if count > 0:
            print(f"🚨 {error_type}: {count} files")
    
    if errors["missing_label"] > 0:
        print("   (Note: 'missing_label' means a .pt file exists but has no matching FILE_ID in the CSV)")
    if errors["index_out_of_bounds"] > 0:
        print("   (Note: 'index_out_of_bounds' is critical and will crash the model!)")

    # --- Node Feature Dimensions (Your original check) ---
    print("\n--- Node Feature Dimensions (for valid files) ---")
    if len(feature_dim_counts) == 1:
        print(f"✅ All {valid_files} valid graphs have the SAME feature dimension.")
    else:
        print(f"🚨 Found {len(feature_dim_counts)} DIFFERENT feature dimensions:")
        for dim, count in sorted(feature_dim_counts.items()):
            print(f"  - Dimension {dim}: {count} files")
    
    # --- Node Count Statistics ---
    print("\n--- Node Counts (for valid files) ---")
    if node_counts:
        node_counts = np.array(node_counts)
        print(f"Graphs have variable node counts (which is OK):")
        print(f"  - Min Nodes: {node_counts.min()}")
        print(f"  - Max Nodes: {node_counts.max()}")
        print(f"  - Avg Nodes: {node_counts.mean():.2f}")
    
    print("\nNext step: Use the 'padding' dataset.py to handle the different feature dimensions.")

if __name__ == "__main__":
    check_dataset_health()