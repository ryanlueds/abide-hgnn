import torch
import glob
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
import pandas as pd
import os
import torch.nn.functional as F
import numpy as np


PATH_GRAPH = "../data/graphs/"
PATH_HYPERGRAPH = "../data/hypergraphs/"
PATH_ABIDE_LABELS = "../abide/Phenotypic_V1_0b_preprocessed1.csv"

# pytorch complains if I do `torch.load` with weights_only=False. It also complains if
# i do weights_only=True, unless I whitelist this garbage
# torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

def normalize_graph(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True)
    return (x - mu) / sd

def calc_pc(X):
    PC = np.corrcoef(X)
    PC = np.nan_to_num(PC, nan=0.0)
    PC = torch.from_numpy(PC).float()
    return PC

class AbideDataset(Dataset):
    def __init__(self, is_hypergraph=True, train=True, split=0.9, split_seed=0, ablation=False):
        self.is_hypergraph = is_hypergraph
        self.dir = PATH_HYPERGRAPH if is_hypergraph else PATH_GRAPH 
        self.ablation = ablation

        all_paths = glob.glob(f'{self.dir}*.pt')

        df = pd.read_csv(PATH_ABIDE_LABELS)
        self.id_to_label_dict = dict(zip(df['FILE_ID'], df['DX_GROUP']))

        rng = torch.Generator().manual_seed(split_seed)
        perm = torch.randperm(len(all_paths), generator=rng).tolist()
        cutoff = int(len(all_paths) * split)
        chosen_idx = perm[:cutoff] if train else perm[cutoff:]
        self.x_paths = [all_paths[i] for i in chosen_idx]


    def __len__(self):
        return len(self.x_paths)


    def __getitem__(self, idx):
        x_path_absolute = self.x_paths[idx]
        x_path_filename = os.path.basename(x_path_absolute)
        file_id = x_path_filename.removesuffix("_hypergraph.pt" if self.is_hypergraph else "_graph.pt")

        x = torch.load(x_path_absolute)
        # x.x = normalize_graph(x.x[:, :self.min_dim])
        x.x = calc_pc(x.x)

        if self.is_hypergraph and self.ablation:
            pairs = x.edge_index.t().tolist()

            buckets = [[] for _ in range(x.num_hyperedges)]
            for n, h in pairs: buckets[h].append(n)

            new_pairs = [[], []]
            num_hyperedges = 0
            for h in range(len(buckets)):
                nodes = sorted(set(buckets[h]))
                if len(nodes) <= 2 and len(nodes) > 0:
                    for node in nodes:
                        new_pairs[0].append(node)
                        new_pairs[1].append(num_hyperedges)
                    num_hyperedges += 1

            x.edge_index = torch.tensor(new_pairs, dtype=torch.long)
            x.num_hyperedges = num_hyperedges

        y = torch.tensor(self.id_to_label_dict[file_id], dtype=torch.long)
        return x, y-1 # y-1 to convert {1, 2} into {0, 1}


# class AbideDataset(Dataset):
#     def __init__(self, is_hypergraph=True, train=True, split=0.9, split_seed=0):
#         self.is_hypergraph = is_hypergraph
#         self.dir = PATH_HYPERGRAPH if is_hypergraph else PATH_GRAPH 

#         all_paths = glob.glob(f'{self.dir}*.pt')

#         df = pd.read_csv(PATH_ABIDE_LABELS)
#         self.id_to_label_dict = dict(zip(df['FILE_ID'], df['DX_GROUP']))

#         rng = torch.Generator().manual_seed(split_seed)
#         perm = torch.randperm(len(all_paths), generator=rng).tolist()
#         cutoff = int(len(all_paths) * split)
#         chosen_idx = perm[:cutoff] if train else perm[cutoff:]
#         self.x_paths = [all_paths[i] for i in chosen_idx]

#         self.max_dim = 0
#         for path in all_paths: self.max_dim = max(self.max_dim, torch.load(path, weights_only=True).x.size(-1))


#     def __len__(self):
#         return len(self.x_paths)


#     def __getitem__(self, idx):
#         x_path_absolute = self.x_paths[idx]
#         x_path_filename = os.path.basename(x_path_absolute)
#         file_id = x_path_filename.removesuffix("_hypergraph.pt" if self.is_hypergraph else "_graph.pt")

#         data = torch.load(x_path_absolute, weights_only=True)
#         cur_dim = int(data.x.size(-1))
#         pad_cols = self.max_dim - cur_dim
#         data.x = normalize_graph(F.pad(data.x, (0, pad_cols), value=0.0))

#         y = torch.tensor(self.id_to_label_dict[file_id], dtype=torch.long)
#         return data, y-1  # convert {1,2} -> {0,1}


# class AbideDataset(Dataset):
#     def __init__(self, is_hypergraph=True, train=True, split=0.9, split_seed=0):
#         self.is_hypergraph = is_hypergraph
#         self.dir = PATH_HYPERGRAPH if is_hypergraph else PATH_GRAPH 

#         all_paths = glob.glob(f'{self.dir}*.pt')

#         df = pd.read_csv(PATH_ABIDE_LABELS)
#         self.id_to_label_dict = dict(zip(df['FILE_ID'], df['DX_GROUP']))

#         rng = torch.Generator().manual_seed(split_seed)
#         perm = torch.randperm(len(all_paths), generator=rng).tolist()
#         cutoff = int(len(all_paths) * split)
#         chosen_idx = perm[:cutoff] if train else perm[cutoff:]
#         self.x_paths = [all_paths[i] for i in chosen_idx]


#     def __len__(self):
#         return len(self.x_paths)


#     def __getitem__(self, idx):
#         x_path_absolute = self.x_paths[idx]
#         x_path_filename = os.path.basename(x_path_absolute)

#         file_id = x_path_filename.removesuffix("_hypergraph.pt" if self.is_hypergraph else "_graph.pt")

#         data = torch.load(x_path_absolute, weights_only=True)
#         mean = data.x.mean(dim=1, keepdim=True)
#         min_val = data.x.min(dim=1).values.unsqueeze(1)
#         max_val = data.x.max(dim=1).values.unsqueeze(1)
#         q25 = data.x.quantile(0.25, dim=1, keepdim=True)
#         q75 = data.x.quantile(0.75, dim=1, keepdim=True)

#         data.x = normalize_graph(torch.cat([mean, min_val, max_val, q25, q75], dim=1))

#         y = torch.tensor(self.id_to_label_dict[file_id], dtype=torch.long)
#         return data, y-1
        

if __name__ == "__main__":
    foo_train = AbideDataset(is_hypergraph=True, train=True, ablation=True)
    print(foo_train.__len__())
    for i in range(5):
        x, y = foo_train[i]
        print(x, y, f"Example {i:3}: {'autism' if y.item() == 0 else 'no autism'}")
    
    # foo_val = AbideDataset(is_hypergraph=True, train=False)
    # print(foo_val.__len__())
    # for i in range(foo_val.__len__()):
    #     x, y = foo_val[i]
    #     print(x, y, f"Example {i:3}: {'autism' if y.item() == 0 else 'no autism'}")

    from torch.utils.data import DataLoader
    
    # no error :D
    train_dataloader = DataLoader(foo_train, batch_size=64, shuffle=True)