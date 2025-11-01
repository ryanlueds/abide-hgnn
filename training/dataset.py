import torch
import glob
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
import pandas as pd
import os
import torch.nn.functional as F


_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PATH_GRAPH = os.path.join(_FILE_DIR, "..", "data", "graphs")
PATH_HYPERGRAPH = os.path.join(_FILE_DIR, "..", "data", "hypergraphs")
PATH_ABIDE_LABELS = os.path.join(_FILE_DIR, "..", "abide", "Phenotypic_V1_0b_preprocessed1.csv")

# pytorch complains if I do `torch.load` with weights_only=False. It also complains if
# i do weights_only=True, unless I whitelist this garbage
torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

def normalize_graph(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True)
    return (x - mu) / sd


class AbideDataset(Dataset):
    def __init__(self, is_hypergraph=True, train=True, split=0.9, split_seed=0):
        self.is_hypergraph = is_hypergraph
        self.dir = PATH_HYPERGRAPH if is_hypergraph else PATH_GRAPH
        all_paths = glob.glob(os.path.join(self.dir, '*.pt'))

        df = pd.read_csv(PATH_ABIDE_LABELS)
        self.id_to_label_dict = dict(zip(df['FILE_ID'], df['DX_GROUP']))

        rng = torch.Generator().manual_seed(split_seed)
        perm = torch.randperm(len(all_paths), generator=rng).tolist()
        cutoff = int(len(all_paths) * split)
        chosen_idx = perm[:cutoff] if train else perm[cutoff:]
        self.x_paths = [all_paths[i] for i in chosen_idx]

        self.min_dim = float('inf')
        for path in all_paths: self.min_dim = min(self.min_dim, torch.load(path, weights_only=True).x.size(-1))


    def __len__(self):
        return len(self.x_paths)


    def __getitem__(self, idx):
        x_path_absolute = self.x_paths[idx]
        x_path_filename = os.path.basename(x_path_absolute)
        file_id = x_path_filename.removesuffix("_hypergraph.pt" if self.is_hypergraph else "_graph.pt")

        x = torch.load(x_path_absolute, weights_only=True)
        x.x = normalize_graph(x.x[:, :self.min_dim])
        y = torch.tensor(self.id_to_label_dict[file_id], dtype=torch.long)
        return x, y-1 # y-1 to convert {1, 2} into {0, 1}


class AbideCorrMatrixDataset(AbideDataset):
    def __init__(self, is_hypergraph=True, train=True, split=0.9, split_seed=0, regularize=False, site_means=None, file_id_to_site_id=None):
        super().__init__(is_hypergraph=is_hypergraph, train=train, split=split, split_seed=split_seed)
        self.regularize = regularize
        self.site_means = site_means
        self.file_id_to_site_id = file_id_to_site_id

        if self.regularize and self.site_means is None:
            if not train:
                print("train=False and regularize=True. This is impossible.")
                return

            df = pd.read_csv(PATH_ABIDE_LABELS)
            self.file_id_to_site_id = dict(zip(df['FILE_ID'], df['SITE_ID']))
            
            site_features = {}
            
            for path in self.x_paths:
                file_id = os.path.basename(path).removesuffix("_hypergraph.pt" if self.is_hypergraph else "_graph.pt")
                
                if file_id not in self.file_id_to_site_id:
                    continue
                
                site = self.file_id_to_site_id[file_id]
                
                data = torch.load(path, weights_only=True)
                timeseries = data.x[:, :self.min_dim]
                corr_matrix = torch.corrcoef(timeseries)
                corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0)
                
                if site not in site_features:
                    site_features[site] = []
                site_features[site].append(corr_matrix)

            self.site_means = {}
            for site, features_list in site_features.items():
                self.site_means[site] = torch.stack(features_list).mean(dim=0)

    def __getitem__(self, idx):
        x_path_absolute = self.x_paths[idx]
        x_path_filename = os.path.basename(x_path_absolute)
        file_id = x_path_filename.removesuffix("_hypergraph.pt" if self.is_hypergraph else "_graph.pt")

        data = torch.load(x_path_absolute, weights_only=True)

        timeseries = data.x[:, :self.min_dim]
        corr_matrix = torch.corrcoef(timeseries)
        corr_matrix = torch.nan_to_num(corr_matrix, nan=0.0)

        if self.regularize and self.site_means is not None:
            site = self.file_id_to_site_id.get(file_id)
            if site and site in self.site_means:
                corr_matrix = corr_matrix - self.site_means[site]

        data.x = normalize_graph(corr_matrix)
        data.x = torch.nan_to_num(data.x, nan=0.0)
        
        y = torch.tensor(self.id_to_label_dict[file_id], dtype=torch.long)
        return data, y-1 # y-1 to convert {1, 2} into {0, 1}

if __name__ == "__main__":
    foo_train = AbideDataset(is_hypergraph=True, train=True)
    print(foo_train.__len__())
    for i in range(foo_train.__len__()):
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
