import torch
import torch.optim as optim
import config as config
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from trainer_dhg import Trainer
from dataset_dhg import AbideDatasetDHG
import dhg


print(config.DEVICE)
torch.manual_seed(config.SEED)
torch.use_deterministic_algorithms(True)

def collate_hg(batch):
    Xs, ys, hgs = zip(*batch)
    return list(Xs), list(ys), list(hgs)

abide_train = AbideDatasetDHG(train=True)
abide_val = AbideDatasetDHG(train=False)

train_dataloader = DataLoader(abide_train, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_hg)
val_dataloader = DataLoader(abide_val, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_hg)

model = dhg.models.HGNNP(in_channels=abide_train[0][0].shape[-1], hid_channels=128, num_classes=2).to(config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=config.LEARN_RATE, weight_decay=config.WEIGHT_DECAY)
trainer = Trainer(config.DEVICE, optimizer, train_dataloader, val_dataloader)
trainer.fit(model)