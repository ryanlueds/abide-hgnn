import torch
import torch.optim as optim
import training_mlp.config as config
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import numpy as np
from training_mlp.model import MLP
from training_mlp.trainer import Trainer
from training_mlp.dataset import AbideDatasetMLP
from training_mlp.plotter import save_cv_plot
import os

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

print(config.DEVICE)
torch.manual_seed(config.SEED)
torch.use_deterministic_algorithms(True)

def write_eval_txt(best_metrics: dict, cv: bool):
    save_dir = os.path.join("results", "mlp")
    os.makedirs(save_dir, exist_ok=True)
    file_name = "cv_eval_summary.txt" if cv else "eval_summary.txt"
    file_path = os.path.join(save_dir, file_name)

    format_map = {
        'acc': ('Accuracy', 100, '%'), 'auroc': ('AUROC', 1, ''),
        'precision': ('Precision', 1, ''), 'recall': ('Recall', 1, '')
    }

    with open(file_path, "w") as f:
        if cv:  
            f.write("=" * 30 + "\nCV AVG EVALUATION RESULTS\n" + "=" * 30 + "\n")
        else:
            f.write("=" * 30 + "\nFINAL EVALUATION RESULTS\n" + "=" * 30 + "\n")
        for k, v in best_metrics.items():
            if k in format_map:
                name, mult, suff = format_map[k]
                if cv:
                    name = f"AVG {name}"
                f.write(f"{name:<11} {v * mult:.4f}{suff}\n")
        f.write("=" * 30 + "\n")


def train_mlp(cv=False):
    if cv:
        full_dataset = AbideDatasetMLP(train=True, split=1.0, split_seed=config.SEED)
        labels = full_dataset.get_strat_labels() 
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

        print(f"Starting Stratified K-Fold (5 Splits)...")

        fold_metrics = []
        # Store history for every fold to aggregate later
        # Structure: {'train_loss': [[fold1_epochs], [fold2_epochs]...], ...}
        cv_history_collection = {
            'train_loss': [], 'train_acc': [], 'train_auroc': [], 'train_precision': [], 'train_recall': [],
            'test_loss': [], 'test_acc': [], 'test_auroc': [], 'test_precision': [], 'test_recall': []
        }

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            print(f"\n--- Fold {fold+1}/5 ---")
            
            train_sub = Subset(full_dataset, train_idx)
            val_sub = Subset(full_dataset, val_idx)
            
            train_dataloader = DataLoader(train_sub, batch_size=config.BATCH_SIZE, shuffle=True)
            val_dataloader = DataLoader(val_sub, batch_size=len(val_sub), shuffle=False)
            
            input_dim = full_dataset[0][0].shape[0]
            model = MLP(in_dim=input_dim, hidden_dim=128).to(config.DEVICE)
            
            optimizer = optim.Adam(model.parameters(), lr=config.LEARN_RATE, weight_decay=config.WEIGHT_DECAY)
            trainer = Trainer(config.DEVICE, optimizer, train_dataloader, val_dataloader)
            
            # UPDATED: Unpack tuple
            best_fold_metrics, fold_history = trainer.fit(model, save_artifacts=False)
            fold_metrics.append(best_fold_metrics)
            
            # Collect history
            for k, v in fold_history.items():
                cv_history_collection[k].append(v)
            
            print(f"    > Fold Result: Acc: {best_fold_metrics['acc']:.4%}, AUROC: {best_fold_metrics['auroc']:.4f}")

        # --- 1. Average Scalar Results (Existing Logic) ---
        print("\n" + "="*30)
        print("FINAL CROSS-VALIDATION RESULTS")
        print("="*30)

        best_avg_metrics = {}
        # Calculate averages as before...
        for key in ['acc', 'auroc', 'precision', 'recall']:
             best_avg_metrics[key] = np.mean([m[key] for m in fold_metrics])

        write_eval_txt(best_avg_metrics, cv=cv)
        
        print(f"Avg Accuracy:  {best_avg_metrics['acc']:.4%}")
        print(f"Avg AUROC:     {best_avg_metrics['auroc']:.4f}")
        
        # --- 2. Generate Aggregated Plots (New Logic) ---
        print("\nGeneratng CV Charts (Median + IQR)...")
        
        metric_pairs = [
            ('Loss', 'train_loss', 'test_loss'),
            ('Accuracy', 'train_acc', 'test_acc'),
            ('AUROC', 'train_auroc', 'test_auroc'),
            ('Precision', 'train_precision', 'test_precision'),
            ('Recall', 'train_recall', 'test_recall'),
        ]

        for pretty_name, train_key, test_key in metric_pairs:
            # Helper to calc stats
            def get_stats(key):
                # Stack folds: shape (n_folds, n_epochs)
                data = np.array(cv_history_collection[key]) 
                return {
                    'median': np.median(data, axis=0),
                    'q1': np.quantile(data, 0.25, axis=0),
                    'q3': np.quantile(data, 0.75, axis=0)
                }

            train_stats = get_stats(train_key)
            test_stats = get_stats(test_key)
            
            save_cv_plot(
                train_stats=train_stats,
                test_stats=test_stats,
                metric_name=pretty_name
            )
        
        print("CV Charts saved to 'charts_cv' directory.")
        print("="*30)

    else:
        # Non-CV Branch
        abide_train = AbideDatasetMLP(train=True, split_seed=config.SEED)
        abide_val = AbideDatasetMLP(train=False, split_seed=config.SEED)

        train_dataloader = DataLoader(abide_train, batch_size=config.BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(abide_val, batch_size=abide_val.__len__(), shuffle=False)

        
        input_dim = abide_train[0][0].shape[0]
        model = MLP(in_dim=input_dim, hidden_dim=128).to(config.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARN_RATE, weight_decay=config.WEIGHT_DECAY)
        trainer = Trainer(config.DEVICE, optimizer, train_dataloader, val_dataloader)
        
        # UPDATED: Unpack tuple (we ignore history here as it's saved inside fit already for single runs)
        best_metrics, _ = trainer.fit(model)

        write_eval_txt(best_metrics, cv=cv)

# # 1. Load the ENTIRE dataset (split=1.0)
# full_dataset = AbideDatasetMLP(train=True, split=1.0, split_seed=config.SEED)

# # 2. Get labels for Stratification
# labels = full_dataset.get_all_labels()

# # 3. Setup Cross Validation
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
# fold_results = []

# print(f"Starting 5-Fold Cross Validation on {len(full_dataset)} samples...")

# for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
#     print(f"\n{'='*20} Fold {fold_idx+1}/5 {'='*20}")
    
#     # 4. Create Subsets
#     train_subset = Subset(full_dataset, train_idx)
#     val_subset = Subset(full_dataset, val_idx)
    
#     train_dataloader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
#     val_dataloader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)
    
#     # 5. Re-initialize Model and Optimizer (Fresh start for every fold)
#     # Get input dimension from the first sample of the dataset
#     input_dim = full_dataset[0][0].shape[0]
    
#     model = MLP(in_dim=input_dim, hidden_dim=128).to(config.DEVICE)
#     optimizer = optim.Adam(model.parameters(), lr=config.LEARN_RATE, weight_decay=config.WEIGHT_DECAY)
    
#     # 6. Train
#     trainer = Trainer(config.DEVICE, optimizer, train_dataloader, val_dataloader)
#     metrics = trainer.fit(model)
#     fold_results.append(metrics)

# # 7. Aggregate and Print Results
# print(f"\n{'='*40}")
# print("CROSS VALIDATION RESULTS (Average over 5 folds)")
# print(f"{'='*40}")

# avg_acc = np.mean([m['acc'] for m in fold_results])
# avg_auroc = np.mean([m['auroc'] for m in fold_results])
# avg_prec = np.mean([m['precision'] for m in fold_results])
# avg_rec = np.mean([m['recall'] for m in fold_results])

# print(f"Avg Accuracy:  {avg_acc:.4%}")
# print(f"Avg AUROC:     {avg_auroc:.4f}")
# print(f"Avg Precision: {avg_prec:.4f}")
# print(f"Avg Recall:    {avg_rec:.4f}")
# print(f"{'='*40}")