import torch
import torch.optim as optim
import training_pyg.config as config
import numpy as np
from torch_geometric.loader import DataLoader
from training_pyg.model import HGNN
from training_pyg.trainer import Trainer
from training_pyg.dataset import AbideDataset
from training_pyg.plotter import save_cv_plot
import os

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

print(config.DEVICE)
torch.manual_seed(config.SEED)
torch.use_deterministic_algorithms(True)

def write_eval_txt(best_metrics: dict, ablation: bool, cv: bool, best_std_metrics=None):
    save_dir = os.path.join("results", "ablation_pyg" if ablation else "pyg")
    os.makedirs(save_dir, exist_ok=True)
    file_name = "cv_eval_summary.txt" if cv else "eval_summary.txt"
    file_path = os.path.join(save_dir, file_name)

    format_map = {
        'acc': ('Accuracy', 1, ''), 'auroc': ('AUROC', 1, ''),
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
                if best_std_metrics is not None:
                    f.write(f"{name:<11} {v * mult:.4f}±{best_std_metrics[k]:.4f}{suff}\n")
                else:
                    f.write(f"{name:<11} {v * mult:.4f}{suff}\n")
        f.write("=" * 30 + "\n")


def train_pyg(cv=False, ablation=False):
    if cv:
        full_dataset = AbideDataset(is_hypergraph=True, train=True, split=1.0, ablation=ablation, split_seed=config.SEED)
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
            
            in_dim = full_dataset[0][0].x.shape[-1]
            model = HGNN(in_dim=in_dim, hidden_dim=64).to(config.DEVICE)
            
            optimizer = optim.Adam(model.parameters(), lr=config.LEARN_RATE, weight_decay=config.WEIGHT_DECAY)
            trainer = Trainer(config.DEVICE, optimizer, train_dataloader, val_dataloader)
            
            # UPDATED: Unpack tuple
            best_fold_metrics, fold_history = trainer.fit(model, save_artifacts=False, ablation=ablation)
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
        best_std_metrics = {}
        # Calculate averages as before...
        for key in ['acc', 'auroc', 'precision', 'recall']:
            values = [m[key] for m in fold_metrics]
            best_avg_metrics[key] = np.mean(values)
            best_std_metrics[key] = np.std(values)

        write_eval_txt(best_avg_metrics, ablation=ablation, cv=cv, best_std_metrics=best_std_metrics)
        
        print(f"Avg Accuracy:  {best_avg_metrics['acc']:.4%} ± {best_std_metrics['acc']:.4%}")
        print(f"Avg AUROC:     {best_avg_metrics['auroc']:.4f} ± {best_std_metrics['auroc']:.4%}")
        
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
                ablation=ablation,
                train_stats=train_stats,
                test_stats=test_stats,
                metric_name=pretty_name
            )
        
        print("CV Charts saved to 'charts_cv' directory.")
        print("="*30)

    else:
        # Non-CV Branch
        abide_train = AbideDataset(is_hypergraph=True, train=True, ablation=ablation, split_seed=config.SEED)
        abide_val = AbideDataset(is_hypergraph=True, train=False, ablation=ablation, split_seed=config.SEED)

        train_dataloader = DataLoader(abide_train, batch_size=config.BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(abide_val, batch_size=abide_val.__len__(), shuffle=False)

        model = HGNN(in_dim=abide_train[0][0].x.shape[-1], hidden_dim=64).to(config.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARN_RATE, weight_decay=config.WEIGHT_DECAY)
        trainer = Trainer(config.DEVICE, optimizer, train_dataloader, val_dataloader)
        
        # UPDATED: Unpack tuple (we ignore history here as it's saved inside fit already for single runs)
        best_metrics, _ = trainer.fit(model, ablation=ablation)

        write_eval_txt(best_metrics, ablation=ablation, cv=cv)