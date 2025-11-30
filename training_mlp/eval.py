import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryPrecision, BinaryRecall
import config as config
from model import MLP
from dataset import AbideDatasetMLP
from torch.utils.data import DataLoader

def evaluate_single_pass(model, loader, device, criterion):
    """
    Performs a single inference pass over the loader and prints metrics.
    """
    model.eval()
    
    # Initialize Metrics
    acc_metric = BinaryAccuracy().to(device)
    auroc_metric = BinaryAUROC().to(device)
    prec_metric = BinaryPrecision().to(device)
    rec_metric = BinaryRecall().to(device)
    
    running_loss = 0.0
    probs_list = []
    targets_list = []

    print(f"Starting evaluation on {device}...")
    
    with torch.no_grad():
        with tqdm(loader, leave=True, desc="Evaluating") as t_dataloader:
            for x, y in t_dataloader:
                x = x.to(device)
                y = y.to(device)
                
                # Forward Pass (Standard MLP input)
                output = model(x)
                
                # Calculate Loss
                loss = criterion(output, y)
                running_loss += loss.item()

                # Store predictions (prob of class 1) and targets
                probs_list.append(torch.softmax(output, dim=1)[:, 1])
                targets_list.append(y.float())

    # Concatenate all batches
    probs = torch.cat(probs_list)
    targets = torch.cat(targets_list)

    # Compute Final Metrics
    final_loss = running_loss / len(loader.dataset)
    final_acc = acc_metric(probs, targets).item()
    final_auroc = auroc_metric(probs, targets).item()
    final_prec = prec_metric(probs, targets).item()
    final_rec = rec_metric(probs, targets).item()

    # Print Results
    print("\n" + "="*30)
    print("FINAL EVALUATION RESULTS")
    print("="*30)
    print(f"Loss:      {final_loss:.4f}")
    print(f"Accuracy:  {final_acc:.4%}")
    print(f"AUROC:     {final_auroc:.4f}")
    print(f"Precision: {final_prec:.4f}")
    print(f"Recall:    {final_rec:.4f}")
    print("="*30 + "\n")

    return final_loss, final_acc, final_auroc, final_prec, final_rec


if __name__ == "__main__":
    # 1. Setup Data
    # Ensure train=False to get the validation/test set
    test_dataset = AbideDatasetMLP(train=False)
    
    # Use standard torch DataLoader
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 2. Setup Model
    # Get input dimension from the first sample (x is index 0)
    input_dim = test_dataset[0][0].shape[0]
    
    # Hidden dim must match training.py (128)
    model = MLP(in_dim=input_dim, hidden_dim=128).to(config.DEVICE)

    # 3. Load Weights
    # Loading 'mlp_model.pt' as saved in your trainer.py
    try:
        model.load_state_dict(torch.load("mlp_model.pt"))
        print("Successfully loaded 'mlp_model.pt'")
    except FileNotFoundError:
        print("Warning: 'mlp_model.pt' not found. Using random weights.")

    # 4. Run Evaluation
    criterion = nn.CrossEntropyLoss()
    evaluate_single_pass(model, test_loader, config.DEVICE, criterion)