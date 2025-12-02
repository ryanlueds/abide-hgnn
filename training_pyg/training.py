import torch
import torch.optim as optim
import config as config
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from model import HGNN
from trainer import Trainer
from dataset import AbideDataset

# NEW IMPORTS
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

print(config.DEVICE)
torch.manual_seed(config.SEED)
torch.use_deterministic_algorithms(True)

# 1. Load ENTIRE dataset so we can fold it ourselves
full_dataset = AbideDataset(is_hypergraph=True, train=True, split=1.0, ablation=True)
labels = full_dataset.get_all_labels() # Uses the helper from dataset.py

# 2. Initialize Stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

print(f"Starting Stratified K-Fold (5 Splits) using parameters from config.py...")
print(f"Params: LR={config.LEARN_RATE}, WD={config.WEIGHT_DECAY}, Batch={config.BATCH_SIZE}")

fold_metrics = []

# 3. Stratified Loop
for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    print(f"\n--- Fold {fold+1}/5 ---")
    
    # Create Subsets
    train_sub = Subset(full_dataset, train_idx)
    val_sub = Subset(full_dataset, val_idx)
    
    train_dataloader = DataLoader(train_sub, batch_size=config.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_sub, batch_size=len(val_sub), shuffle=False)
    
    # Init Model & Optimizer (Using Config constants)
    in_dim = full_dataset[0][0].x.shape[-1]
    model = HGNN(
        in_dim=in_dim, 
        hidden_dim=64 # Or config.HIDDEN_DIM if you added it there
    ).to(config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARN_RATE, weight_decay=config.WEIGHT_DECAY)
    trainer = Trainer(config.DEVICE, optimizer, train_dataloader, val_dataloader)
    
    # Run Training
    # save_artifacts=False prevents overwriting plots 5 times
    # We capture the BEST metrics from this specific fold
    best_fold_metrics = trainer.fit(model, save_artifacts=False) 
    fold_metrics.append(best_fold_metrics)
    
    print(f"    > Fold Result: Acc: {best_fold_metrics['acc']:.4%}, AUROC: {best_fold_metrics['auroc']:.4f}")

# 4. Average Results
print("\n" + "="*30)
print("FINAL CROSS-VALIDATION RESULTS")
print("="*30)

avg_acc = np.mean([m['acc'] for m in fold_metrics])
avg_auroc = np.mean([m['auroc'] for m in fold_metrics])
avg_prec = np.mean([m['precision'] for m in fold_metrics])
avg_rec = np.mean([m['recall'] for m in fold_metrics])

print(f"Avg Accuracy:  {avg_acc:.4%}")
print(f"Avg AUROC:     {avg_auroc:.4f}")
print(f"Avg Precision: {avg_prec:.4f}")
print(f"Avg Recall:    {avg_rec:.4f}")
print("="*30)