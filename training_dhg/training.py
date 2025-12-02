import torch
import torch.optim as optim
import config as config
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from trainer_dhg import Trainer
from dataset_dhg import AbideDatasetDHG
import dhg
import numpy as np

print(config.DEVICE)
torch.manual_seed(config.SEED)
torch.use_deterministic_algorithms(True)

def collate_hg(batch):
    Xs, ys, hgs = zip(*batch)
    return list(Xs), list(ys), list(hgs)

# 1. Load the ENTIRE dataset (split=1.0 means train gets everything)
full_dataset = AbideDatasetDHG(train=True, split=1.0, split_seed=config.SEED)

# 2. Get labels for Stratification
labels = full_dataset.get_all_labels()

# 3. Setup Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
fold_results = []

print(f"Starting 5-Fold Cross Validation on {len(full_dataset)} samples...")

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"\n{'='*20} Fold {fold_idx+1}/5 {'='*20}")
    
    # 4. Create Subsets
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_hg)
    val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_hg)
    
    # 5. Re-initialize Model and Optimizer (Fresh start for every fold)
    # Note: Accessing first element of subset to get input dim: full_dataset[0][0]
    model = dhg.models.UniSAGE(in_channels=full_dataset[0][0].shape[-1], hid_channels=128, num_classes=2).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARN_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # 6. Train
    trainer = Trainer(config.DEVICE, optimizer, train_loader, val_loader)
    metrics = trainer.fit(model)
    fold_results.append(metrics)

# 7. Aggregate and Print Results
print(f"\n{'='*40}")
print("CROSS VALIDATION RESULTS (Average over 5 folds)")
print(f"{'='*40}")

avg_acc = np.mean([m['acc'] for m in fold_results])
avg_auroc = np.mean([m['auroc'] for m in fold_results])
avg_prec = np.mean([m['precision'] for m in fold_results])
avg_rec = np.mean([m['recall'] for m in fold_results])

print(f"Avg Accuracy:  {avg_acc:.4%}")
print(f"Avg AUROC:     {avg_auroc:.4f}")
print(f"Avg Precision: {avg_prec:.4f}")
print(f"Avg Recall:    {avg_rec:.4f}")
print(f"{'='*40}")