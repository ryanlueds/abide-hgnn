import torch
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryPrecision, BinaryRecall
import config as config
from model import HGNN
from dataset import AbideDataset
from torch_geometric.loader import DataLoader

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
            for data in t_dataloader:
                # Handle Tuple (Batch, Label) format to match Trainer logic
                # data[0] is the PyG Batch object, data[1] is the Label tensor
                x_graph = data[0].to(device)
                y = data[1].to(device)
                
                # Unpack PyG Data object components
                x_features = x_graph.x
                hedge = x_graph.edge_index
                batch_vector = x_graph.batch

                # Forward Pass
                output = model(x_features, hedge, batch_vector)
                
                # Calculate Loss
                loss = criterion(output, y)
                running_loss += loss.item()

                # Store predictions (prob of class 1) and targets
                probs_list.append(torch.softmax(output, dim=1)[:, 1])
                targets_list.append(y)

    # Concatenate all batches
    probs = torch.cat(probs_list)
    targets = torch.cat(targets_list)

    # Compute Final Metrics
    final_loss = running_loss / len(loader)
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
    test_dataset = AbideDataset(is_hypergraph=True, train=False)
    
    # Use full length as batch size (Full Batch Inference)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    # 2. Setup Model
    # Access [0][0] because dataset returns tuple (Data, Label)
    sample_data = test_dataset[0][0] 
    in_dim = sample_data.x.shape[-1]
    
    model = HGNN(in_dim=in_dim, hidden_dim=64).to(config.DEVICE)

    # 3. Load Weights
    # Load the best model saved during training
    try:
        model.load_state_dict(torch.load("pyg_model.pt"))
        print("Successfully loaded 'pyg_model.pt'")
    except FileNotFoundError:
        print("Warning: 'pyg_model.pt' not found. Using random weights.")

    # 4. Run Evaluation
    criterion = nn.CrossEntropyLoss()
    evaluate_single_pass(model, test_loader, config.DEVICE, criterion)