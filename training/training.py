import torch
import torch.optim as optim
import config as config
import pandas as pd
from torch_geometric.loader import DataLoader
from model import HGNN
from trainer import Trainer
from dataset import AbideDataset, AbideCorrMatrixDataset


print(config.DEVICE)
torch.manual_seed(config.SEED)

abide_train = AbideCorrMatrixDataset(is_hypergraph=True, train=True, regularize=True)
abide_val = AbideCorrMatrixDataset(is_hypergraph=True, train=False, regularize=True, 
                                   site_means=abide_train.site_means, 
                                   file_id_to_site_id=abide_train.file_id_to_site_id)

train_dataloader = DataLoader(abide_train, batch_size=config.BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(abide_val, batch_size=abide_val.__len__(), shuffle=False)

model = HGNN(in_dim=abide_train[0][0].x.shape[-1], hidden_dim=32).to(config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=config.LEARN_RATE, weight_decay=config.WEIGHT_DECAY)
trainer = Trainer(config.DEVICE, optimizer, train_dataloader, val_dataloader)
trainer.fit(model)
