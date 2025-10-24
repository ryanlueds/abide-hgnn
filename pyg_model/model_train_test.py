import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np

# Import our custom modules
from model_architecture import HGNN_Graph

# To access dataset from child directory
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, os.pardir)
sys.path.append(parent_dir)
from dataset import AbideDataset, ENGINEERED_FEATURE_DIM

# --- Configuration ---

# Check for CUDA, then MPS (Apple Silicon), then fallback to CPU
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 50
HIDDEN_CHANNELS = 64
DROPOUT = 0.25
TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation
RANDOM_STATE = 42

def collate_fn_skip_none(batch):
    """
    Custom collate function to filter out None values
    (e.g., from failed data loading).
    """
    batch = [item for item in batch if item is not None]
    return DataLoader.collate(batch) if batch else None

def train_epoch(model, loader, optimizer, criterion):
    """Runs one training epoch."""
    model.train()
    total_loss = 0
    for batch in loader:
        if batch is None: continue  # Skip empty batches
        batch = batch.to(DEVICE)
        
        optimizer.zero_grad()
        out = model(batch.x, batch.hyperedge_index, batch.batch)
        loss = criterion(out, batch.y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        
    return total_loss / len(loader.dataset)

def test_epoch(model, loader, criterion):
    """Runs one validation/test epoch."""
    model.eval()
    total_loss = 0
    
    # Initialize metrics
    acc_metric = BinaryAccuracy().to(DEVICE)
    auc_metric = BinaryAUROC().to(DEVICE)
    
    with torch.no_grad():
        for batch in loader:
            if batch is None: continue
            batch = batch.to(DEVICE)
            
            out = model(batch.x, batch.hyperedge_index, batch.batch)
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            
            # Update metrics
            probs = torch.sigmoid(out)
            acc_metric.update(probs, batch.y)
            auc_metric.update(probs, batch.y)
            
    avg_loss = total_loss / len(loader.dataset)
    accuracy = acc_metric.compute().item()
    auc = auc_metric.compute().item()
    
    return avg_loss, accuracy, auc

def main():
    print(f"Using device: {DEVICE}")

    # Load dataset
    dataset = AbideDataset(is_hypergraph=True)
    
    # Scan dataset for labels and class imbalance
    print("Scanning dataset to gather labels for stratification...")
    all_labels = []
    valid_indices = []
    
    total_samples = len(dataset)
    for i in range(total_samples):
        data = dataset[i]
        if data is not None:
            all_labels.append(data.y.item())
            valid_indices.append(i)
        
        # Simple print progress
        if (i + 1) % 100 == 0 or (i + 1) == total_samples:
            print(f"  ...scanned {i+1}/{total_samples} samples", end='\r')

    print("\nScan complete.") # Move to next line after scan

    total_valid = len(all_labels)
    class_1_count = np.sum(all_labels)
    class_0_count = total_valid - class_1_count

    print("\n--- Dataset Class Imbalance ---")
    print(f"Total valid samples: {total_valid} (out of {len(dataset)})")
    if total_valid > 0:
        print(f"  Class 0 (Control): {int(class_0_count)} ({class_0_count/total_valid*100:.2f}%)")
        print(f"  Class 1 (Autism):  {int(class_1_count)} ({class_1_count/total_valid*100:.2f}%)")
    print("---------------------------------\n")

    # Create Stratified Train/Val Splits
    train_indices, val_indices = train_test_split(
        valid_indices,  # List of indices to split
        train_size=TRAIN_SPLIT,
        stratify=all_labels, # Use the labels for stratification
        random_state=RANDOM_STATE
    )

    # Create Subset datasets from the indices
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn_skip_none
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn_skip_none
    )

    # Initialize Model, Loss, Optimizer
    model = HGNN_Graph(
        in_channels=ENGINEERED_FEATURE_DIM,
        hidden_channels=HIDDEN_CHANNELS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=5e-5    
    )

    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max',      # We want to maximize AUC
        factor=0.1,      # Reduce LR by 10x
        patience=10,      # Wait 10 epochs with no improvement
        verbose=True
    )
    
    print("\n--- Starting Training ---")
    print(f"Model: {model}")
    print(f"Input features: {ENGINEERED_FEATURE_DIM}, Hidden: {HIDDEN_CHANNELS}")

    # Training Loop
    best_val_auc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_auc = test_epoch(model, val_loader, criterion)
        scheduler.step(val_auc)
        
        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'pyg_model/best_model.pt')
            print(f"  -> New best model saved with AUC: {best_val_auc:.4f}")

    print("--- Training Complete ---")
    print(f"Best validation AUC achieved: {best_val_auc:.4f}")

if __name__ == "__main__":
    main()