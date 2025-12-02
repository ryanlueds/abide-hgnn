import torch
import glob
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
import pandas as pd
import os
import torch.nn.functional as F
import dhg
import numpy as np
from sklearn.model_selection import train_test_split

PATH_GRAPH = "../data/graphs/"
PATH_HYPERGRAPH = "../data/hypergraphs/"
PATH_ABIDE_LABELS = "../abide/Phenotypic_V1_0b_preprocessed1.csv"

# torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

def normalize_graph(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True)
    return (x - mu) / sd


# class AbideDatasetDHG(Dataset):
#     def __init__(self, train=True, split=0.9, split_seed=0):
#         self.dir = PATH_HYPERGRAPH

#         all_paths = glob.glob(f"{self.dir}*.pt")

#         df = pd.read_csv(PATH_ABIDE_LABELS)
#         self.id_to_label_dict = dict(zip(df["FILE_ID"], df["DX_GROUP"]))

#         rng = torch.Generator().manual_seed(split_seed)
#         perm = torch.randperm(len(all_paths), generator=rng).tolist()
#         cutoff = int(len(all_paths) * split)
#         chosen_idx = perm[:cutoff] if train else perm[cutoff:]
#         self.x_paths = [all_paths[i] for i in chosen_idx]

#         # Identify minimum time series length
#         self.min_dim = float("inf")
#         for path in all_paths: self.min_dim = min(self.min_dim, torch.load(path, weights_only=False).x.size(-1))


#     def __len__(self):
#         return len(self.x_paths)


#     def __getitem__(self, idx):
#         x_path_absolute = self.x_paths[idx]
#         x_path_filename = os.path.basename(x_path_absolute)
#         file_id = x_path_filename.removesuffix("_hypergraph.pt")

#         x = torch.load(x_path_absolute, weights_only=False)
#         # Truncate to the minimium time series length
#         x.x = normalize_graph(x.x[:, : self.min_dim])

#         pairs = x.edge_index.t().tolist()
#         num_nodes = x.x.size(0)
#         buckets = [[] for _ in range(x.num_hyperedges)]
#         for n, h in pairs: buckets[h].append(n)
#         hedges = [tuple(sorted(set(ns))) for ns in buckets if len(ns) > 0]
#         hg = dhg.Hypergraph(num_nodes, hedges)
#         y = torch.tensor(self.id_to_label_dict[file_id], dtype=torch.long)
#         return x.x, y-1, hg


def calc_pc(X):
    PC = np.corrcoef(X)
    PC = np.nan_to_num(PC, nan=0.0)
    PC = torch.from_numpy(PC).float()
    return PC

class AbideDatasetDHG(Dataset):
    def __init__(self, train=True, split=0.8, split_seed=0, ablation=False):
        self.dir = PATH_HYPERGRAPH
        self.ablation = ablation

        all_paths = glob.glob(f"{self.dir}*.pt")
        all_paths.sort()
        all_paths = [p for p in all_paths if not os.path.basename(p).startswith('.')]

        df = pd.read_csv(PATH_ABIDE_LABELS)
        self.id_to_label_dict = dict(zip(df["FILE_ID"], df["DX_GROUP"]))
        self.id_to_site_dict = dict(zip(df["FILE_ID"], df["SITE_ID"]))

        labels = []
        sites = []
        valid_paths = []

        for p in all_paths:
            x_path_filename = os.path.basename(p)
            file_id = x_path_filename.removesuffix("_hypergraph.pt")
            
            # Handle potential type mismatch (CSV might have ints, filenames are strings)
            label = self.id_to_label_dict.get(file_id) or self.id_to_label_dict.get(int(file_id) if file_id.isdigit() else None)
            site = self.id_to_site_dict.get(file_id)
            
            if label is not None and site is not None:
                labels.append(label)
                sites.append(site)
                valid_paths.append(p)
            else:
                print(f"Warning: Label/Site not found for {file_id}, excluding from split.")

        strat_labels = [loc + str(lab) for lab, loc in zip(labels, sites)]
        train_paths, test_paths = train_test_split(
            valid_paths,
            train_size=split,
            stratify=strat_labels,
            random_state=split_seed
        )

        self.x_paths = train_paths if train else test_paths


    def __len__(self):
        return len(self.x_paths)
    
    def get_all_labels(self):
        labels = []
        for path in self.x_paths:
            x_path_filename = os.path.basename(path)
            file_id = x_path_filename.removesuffix("_hypergraph.pt")
            # 2-y to match the __getitem__ logic {1=+, 2=-} -> {0=-, 1=+}
            labels.append(2 - self.id_to_label_dict[file_id])
        return labels


    def __getitem__(self, idx):
        x_path_absolute = self.x_paths[idx]
        x_path_filename = os.path.basename(x_path_absolute)
        file_id = x_path_filename.removesuffix("_hypergraph.pt")

        x = torch.load(x_path_absolute, weights_only=False)
        x.x = calc_pc(x.x)
        # # Truncate to the minimium time series length
        # x.x = normalize_graph(x.x[:, : self.min_dim])

        pairs = x.edge_index.t().tolist()
        num_nodes = x.x.size(0)

        buckets = [[] for _ in range(x.num_hyperedges)]
        for n, h in pairs: buckets[h].append(n)
        hedges = [tuple(sorted(set(ns))) for ns in buckets if len(ns) > 0]
        if self.ablation: hedges = [h for h in hedges if len(h) <= 2]
        hg = dhg.Hypergraph(num_nodes, hedges)

        y = torch.tensor(self.id_to_label_dict[file_id], dtype=torch.long)
        return x.x, 2-y, hg # 2-y to convert {1=+, 2=-} into {0=-, 1=+}


if __name__ == "__main__":
    foo_train = AbideDatasetDHG(train=True)
    for i in range(10):
        x, y, hg = foo_train[i]
        print(x.shape, y, hg, f"Example {i:3}: {'autism' if y.item() == 0 else 'no autism'}")

    from torch.utils.data import DataLoader
    
    train_dataloader = DataLoader(foo_train, batch_size=64, shuffle=True)
