import torch
import glob
from torch.utils.data import Dataset
from torch_geometric.data import Data
# We need to import the geometric DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
import pandas as pd
import os


PATH_GRAPH = "./data/graphs/"
PATH_HYPERGRAPH = "./data/hypergraphs/"
PATH_ABIDE_LABELS = "./abide/Phenotypic_V1_0b_preprocessed1.csv"

# This will be our new fixed feature dimension
ENGINEERED_FEATURE_DIM = 5

# pytorch complains if I do `torch.load` with weights_only=False. It also complains if
# i do weights_only=True, unless I whitelist this garbage
torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

class AbideDataset(Dataset):
    
    # Engineer fixed size node feature vector from time series data
    def _engineer_features(self, ts_tensor):
        """
        Helper method to compute statistical features from a time series tensor.
        Input: ts_tensor (torch.Tensor): Shape [num_nodes, variable_time_len]
        Output: (torch.Tensor): Shape [num_nodes, ENGINEERED_FEATURE_DIM]
        """
        
        # Ensure tensor is float for calculations
        ts_tensor = ts_tensor.float()
        
        # Compute features along the time dimension (dim=1)
        ts_mean = torch.mean(ts_tensor, dim=1, keepdim=True)
        ts_std = torch.std(ts_tensor, dim=1, keepdim=True)
        ts_max = torch.max(ts_tensor, dim=1, keepdim=True).values
        ts_min = torch.min(ts_tensor, dim=1, keepdim=True).values
        ts_median = torch.median(ts_tensor, dim=1, keepdim=True).values
        
        # Concatenate features to create the new feature vector
        feature_vector = torch.cat([
            ts_mean,
            ts_std,
            ts_max,
            ts_min,
            ts_median
        ], dim=1)
        
        # Handle potential NaNs from std dev (if time series is constant)
        return torch.nan_to_num(feature_vector, nan=0.0)

    def __init__(self, is_hypergraph=False):
        self.is_hypergraph = is_hypergraph
        self.dir = PATH_HYPERGRAPH if is_hypergraph else PATH_GRAPH 

        self.x_paths = glob.glob(f'{self.dir}*.pt')

        df = pd.read_csv(PATH_ABIDE_LABELS)
        # Convert DX_GROUP (1 and 2) to (0 and 1) for binary classification
        df['DX_GROUP'] = df['DX_GROUP'].apply(lambda x: 1.0 if x == 1 else 0.0)
        self.id_to_label_dict = dict(zip(df['FILE_ID'], df['DX_GROUP']))

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        x_path_absolute = self.x_paths[idx]
        x_path_filename = os.path.basename(x_path_absolute)
        file_id = x_path_filename.removesuffix("_hypergraph.pt" if self.is_hypergraph else "_graph.pt")

        # 'data' is the loaded PyG Data object
        try:
            data = torch.load(x_path_absolute, weights_only=True)
        except Exception as e:
            print(f"Error loading {file_id}: {e}. Skipping.")
            return None # The DataLoader will skip None items
        
        # Check for label
        if file_id not in self.id_to_label_dict:
            print(f"Missing label for {file_id}. Skipping.")
            return None
        
        # Check for 'x'
        if not hasattr(data, 'x') or data.x is None:
            print(f"Missing 'x' for {file_id}. Skipping.")
            return None

        # Apply feature engineering for fixed length node feature vectors
        data.x = self._engineer_features(data.x)
        
        # Rename edge_index to hyperedge_index
        if hasattr(data, 'edge_index'):
            data.hyperedge_index = data.edge_index
            del data.edge_index
        elif not hasattr(data, 'hyperedge_index'):
             print(f"Missing 'edge_index' or 'hyperedge_index' for {file_id}. Skipping.")
             return None

        # Attach label and num_nodes
        label_val = self.id_to_label_dict[file_id]
        data.y = torch.tensor([label_val], dtype=torch.float)
        
        if not hasattr(data, 'num_nodes'):
            data.num_nodes = data.x.shape[0]
        
        # Return the Data object
        return data
        

if __name__ == "__main__":    
    print("Testing Hypergraph Dataset and DataLoader:")
    
    hypergraph_dataset = AbideDataset(is_hypergraph=True)
    
    # Test __getitem__
    print(f"Dataset size: {len(hypergraph_dataset)}")
    
    # Filter out None from __getitem__
    # Need a collate_fn to handle potential Nones
    def collate_fn_skip_none(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None
        # Use the default collate function from DataLoader
        return DataLoader.collate(batch)

    # We must get a valid first item to print
    first_data = None
    for i in range(len(hypergraph_dataset)):
        first_data = hypergraph_dataset[i]
        if first_data is not None:
            print(f"First valid data object:\n{first_data}")
            print(f"Feature shape: {first_data.x.shape}") # Should be [nodes, 5]
            print(f"Label for first data: {first_data.y.item()} ({'autism' if first_data.y.item() == 1.0 else 'control'})")
            break

    print("\nTesting DataLoader...")
    # This will now use the PyG collate function correctly
    train_loader = DataLoader(
        hypergraph_dataset, 
        batch_size=4, 
        shuffle=True,
        collate_fn=collate_fn_skip_none # Use our custom collate
    )
    
    # Get one batch
    batch = next(iter(train_loader))
    
    if batch:
        print(f"\nBatch object:\n{batch}")
        print(f"Batch 'x' shape: {batch.x.shape}") # Should be [total_nodes, 5]
        
        # Check if hyperedge_index exists
        if hasattr(batch, 'hyperedge_index'):
            print(f"Batch 'hyperedge_index' shape: {batch.hyperedge_index.shape}")
        else:
            print("Batch missing 'hyperedge_index'!")
            
        print(f"Batch 'y' labels: {batch.y.squeeze()}")
        print(f"Batch 'batch' vector (maps nodes to graphs): {batch.batch}")
        print(f"\nIN_CHANNELS for your model should be: {ENGINEERED_FEATURE_DIM}")
    else:
        print("DataLoader returned an empty batch. Check data integrity.")