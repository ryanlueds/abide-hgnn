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

PATH_GRAPH = "../data/graphs/"
PATH_HYPERGRAPH = "../data/hypergraphs/"
PATH_ABIDE_LABELS = "../abide/Phenotypic_V1_0b_preprocessed1.csv"

torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

def normalize_graph(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True)
    return (x - mu) / sd


class AbideDatasetDHG(Dataset):
    def __init__(self, train=True, split=0.9, split_seed=0):
        self.dir = PATH_HYPERGRAPH

        all_paths = glob.glob(f"{self.dir}*.pt")

        df = pd.read_csv(PATH_ABIDE_LABELS)
        self.id_to_label_dict = dict(zip(df["FILE_ID"], df["DX_GROUP"]))

        rng = torch.Generator().manual_seed(split_seed)
        perm = torch.randperm(len(all_paths), generator=rng).tolist()
        cutoff = int(len(all_paths) * split)
        chosen_idx = perm[:cutoff] if train else perm[cutoff:]
        self.x_paths = [all_paths[i] for i in chosen_idx]

        self.min_dim = float("inf")
        for path in all_paths: self.min_dim = min(self.min_dim, torch.load(path, weights_only=True).x.size(-1))


    def __len__(self):
        return len(self.x_paths)


    def __getitem__(self, idx):
        x_path_absolute = self.x_paths[idx]
        x_path_filename = os.path.basename(x_path_absolute)
        file_id = x_path_filename.removesuffix("_hypergraph.pt")

        x = torch.load(x_path_absolute, weights_only=True)
        x.x = normalize_graph(x.x[:, : self.min_dim])

        pairs = x.edge_index.t().tolist()
        num_nodes = x.x.size(0)
        buckets = [[] for _ in range(x.num_hyperedges)]
        for n, h in pairs: buckets[h].append(n)
        hedges = [tuple(sorted(set(ns))) for ns in buckets if len(ns) > 0]
        hg = dhg.Hypergraph(num_nodes, hedges)

        y = torch.tensor(self.id_to_label_dict[file_id], dtype=torch.long)
        return x.x, y-1, hg


if __name__ == "__main__":
    foo_train = AbideDatasetDHG(train=True)
    for i in range(10):
        x, y, hg = foo_train[i]
        print(x.shape, y, hg, f"Example {i:3}: {'autism' if y.item() == 0 else 'no autism'}")

    from torch.utils.data import DataLoader
    
    train_dataloader = DataLoader(foo_train, batch_size=64, shuffle=True)
