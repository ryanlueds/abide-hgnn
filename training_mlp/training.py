import torch
import torch.optim as optim
import config as config
from torch.utils.data import DataLoader
from model import MLP
from trainer import Trainer
from dataset import AbideDatasetMLP


torch.manual_seed(config.SEED)

abide_train = AbideDatasetMLP(train=True)
abide_val = AbideDatasetMLP(train=False)

train_dataloader = DataLoader(abide_train, batch_size=config.BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(abide_val, batch_size=config.BATCH_SIZE, shuffle=False)

input_dim = abide_train[0][0].shape[0]

model = MLP(in_dim=input_dim, hidden_dim=128).to(config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=config.LEARN_RATE, weight_decay=config.WEIGHT_DECAY)
trainer = Trainer(config.DEVICE, optimizer, train_dataloader, val_dataloader)
trainer.fit(model)
