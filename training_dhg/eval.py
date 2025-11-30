import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryPrecision, BinaryRecall
import config as config
import dhg
from dataset_dhg import AbideDatasetDHG
from torch.utils.data import DataLoader

def collate_hg(batch):
    Xs, ys, hgs = zip(*batch)
    return list(Xs), list(ys), list(hgs)

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
            # Loader yields lists of graphs/labels/hypergraphs due to collate_hg
            for xs, ys, hgs in t_dataloader:
                # Process each graph in the batch individually (as done in trainer_dhg.py)
                for x, y, hg in zip(xs, ys, hgs):
                    x = x.to(device)
                    y = y.to(device).unsqueeze(0) # Add batch dimension [1]
                    hg = hg.to(device)
                    
                    # Forward Pass
                    # HGNNP returns node embeddings, mean pool for graph classification
                    output = model(x, hg).mean(dim=0, keepdim=True)
                    
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
    test_dataset = AbideDatasetDHG(train=False)
    
    # Use DataLoader with the custom collate function required for DHG
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_hg
    )
    
    # 2. Setup Model
    # Get input dimension from the first sample
    sample_x, _, _ = test_dataset[0]
    in_channels = sample_x.shape[-1]
    
    # Model definition matches training.py: HGNNP with hid_channels=128
    model = dhg.models.HGNNP(in_channels=in_channels, hid_channels=128, num_classes=2).to(config.DEVICE)

    # 3. Load Weights
    # Loading 'dhg_model.pt' as saved in trainer_dhg.py
    try:
        model.load_state_dict(torch.load("dhg_hgnnp_model.pt"))
        print("Successfully loaded 'dhg_model.pt'")
    except FileNotFoundError:
        print("Warning: 'dhg_model.pt' not found. Using random weights.")

    # 4. Run Evaluation
    criterion = nn.CrossEntropyLoss()
    evaluate_single_pass(model, test_loader, config.DEVICE, criterion)