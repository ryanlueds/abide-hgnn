import torch
import glob
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
import pandas as pd
import os


PATH_GRAPH = "./data/graphs/"
PATH_HYPERGRAPH = "./data/hypergraphs/"
PATH_ABIDE_LABELS = "./abide/Phenotypic_V1_0b_preprocessed1.csv"

# pytorch complains if I do `torch.load` with weights_only=False. It also complains if
# i do weights_only=True, unless I whitelist this garbage
torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

class AbideDataset(Dataset):
    def __init__(self, is_hypergraph=False):
        self.is_hypergraph = is_hypergraph
        self.dir = PATH_HYPERGRAPH if is_hypergraph else PATH_GRAPH 

        self.x_paths = glob.glob(f'{self.dir}*.pt')

        df = pd.read_csv(PATH_ABIDE_LABELS)
        self.id_to_label_dict = dict(zip(df['FILE_ID'], df['DX_GROUP']))

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        x_path_absolute = self.x_paths[idx]
        x_path_filename = os.path.basename(x_path_absolute)
        file_id = x_path_filename.removesuffix("_hypergraph.pt" if self.is_hypergraph else "_graph.pt")

        x = torch.load(x_path_absolute, weights_only=True)
        y = torch.tensor(self.id_to_label_dict[file_id], dtype=torch.long)
        return x, y
        

if __name__ == "__main__":
    foo = AbideDataset(is_hypergraph=True)
    print("Hypergraph data labels test:")
    for i in range(10):
        _, y = foo[i]
        print(f"{i:3}: {'autism' if y.item() == 1 else 'no autism'}")

    print()

    foo = AbideDataset()
    print("Graph data labels test:")
    for i in range(10):
        _, y = foo[i]
        print(f"{i:3}: {'autism' if y.item() == 1 else 'no autism'}")



    from torch.utils.data import DataLoader
    
    # no error :D
    train_dataloader = DataLoader(foo, batch_size=64, shuffle=True)

