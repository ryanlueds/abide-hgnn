import torch
import glob
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

PATH_GRAPH = "../data/graphs/"
PATH_HYPERGRAPH = "../data/hypergraphs/"
PATH_ABIDE_LABELS = "../abide/Phenotypic_V1_0b_preprocessed1.csv"

def normalize_graph(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True)
    return (x - mu) / sd

def calc_pc(X):
    PC = np.corrcoef(X)
    PC = np.nan_to_num(PC, nan=0.0)
    PC = torch.from_numpy(PC).float()
    return PC

class AbideDatasetMLP(Dataset):
    def __init__(self, train=True, split=0.9, split_seed=0):
        self.dir = PATH_HYPERGRAPH 

        all_paths = glob.glob(f'{self.dir}*.pt')
        all_paths.sort()

        df = pd.read_csv(PATH_ABIDE_LABELS)
        self.id_to_label_dict = dict(zip(df['FILE_ID'], df['DX_GROUP']))

        rng = torch.Generator().manual_seed(split_seed)
        perm = torch.randperm(len(all_paths), generator=rng).tolist()
        cutoff = int(len(all_paths) * split)
        chosen_idx = perm[:cutoff] if train else perm[cutoff:]
        self.x_paths = [all_paths[i] for i in chosen_idx]

        self.min_dim = float('inf')
        for path in all_paths: 
            self.min_dim = min(self.min_dim, torch.load(path, weights_only=False).x.size(-1))

    def __len__(self):
        return len(self.x_paths)
    
    def get_all_labels(self):
        labels = []
        for path in self.x_paths:
            x_path_filename = os.path.basename(path)
            file_id = x_path_filename.removesuffix("_hypergraph.pt")
            # 2-y to convert {1=+, 2=-} into {0=-, 1=+} (Matches __getitem__)
            labels.append(2 - self.id_to_label_dict[file_id])
        return labels

    def __getitem__(self, idx):
        x_path_absolute = self.x_paths[idx]
        x_path_filename = os.path.basename(x_path_absolute)
        file_id = x_path_filename.removesuffix("_hypergraph.pt")

        data = torch.load(x_path_absolute, weights_only=False)
        # truncate time series
        # x_tensor = data.x[:, :self.min_dim]
        
        x_tensor = calc_pc(data.x)

        x_flat = x_tensor.flatten()

        y = torch.tensor(self.id_to_label_dict[file_id], dtype=torch.long)
        return x_flat, 2-y # 2-y to convert {1=+, 2=-} into {0=-, 1=+}

if __name__ == "__main__":
    foo_train = AbideDatasetMLP(train=True)
    print(f"Dataset length: {foo_train.__len__()}")
    if len(foo_train) > 0:
        x, y = foo_train[0]
        print(f"sample shape: {x.shape}, label: {y}, class: {'autism' if y.item() == 0 else 'no autism'}")

    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(foo_train, batch_size=64, shuffle=True)
    for batch_x, batch_y in train_dataloader:
        print(f"batch shape: {batch_x.shape}")
        break
