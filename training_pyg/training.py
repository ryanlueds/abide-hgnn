import torch
import torch.optim as optim
import config as config
import pandas as pd
from torch_geometric.loader import DataLoader
from model import HGNN
from trainer import Trainer
from dataset import AbideDataset


print(config.DEVICE)
torch.manual_seed(config.SEED)
torch.use_deterministic_algorithms(True)

abide_train = AbideDataset(is_hypergraph=True, train=True)
abide_val = AbideDataset(is_hypergraph=True, train=False)

train_dataloader = DataLoader(abide_train, batch_size=config.BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(abide_val, batch_size=abide_val.__len__(), shuffle=False)

model = HGNN(in_dim=abide_train[0][0].x.shape[-1], hidden_dim=64).to(config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=config.LEARN_RATE, weight_decay=config.WEIGHT_DECAY)
trainer = Trainer(config.DEVICE, optimizer, train_dataloader, val_dataloader)
trainer.fit(model)