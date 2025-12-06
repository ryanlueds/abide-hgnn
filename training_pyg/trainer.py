import torch
import torch.nn as nn
from tqdm import tqdm
import os
import training_pyg.config as config
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryPrecision, BinaryRecall
from training_pyg.plotter import save_plot

class Trainer(object):

    def __init__(self, device, optimizer, train_loader, test_loader, criterion=nn.CrossEntropyLoss()):
        self.device = device
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion


    def fit(self, model, save_artifacts=True, ablation=False):
        history = {
            'train_loss': [], 'train_acc': [], 'train_auroc': [], 'train_precision': [], 'train_recall': [],
            'test_loss': [], 'test_acc': [], 'test_auroc': [], 'test_precision': [], 'test_recall': []
        }

        num_epochs = config.EPOCHS
        best_test_auroc = -1.0
        best_metrics = {}

        for epoch in range(num_epochs):
            train_loss, train_acc, train_auroc, train_precision, train_recall = self.train_step(model, epoch)
            test_loss, test_acc, test_auroc, test_precision, test_recall = self.testing_step(model)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_auroc'].append(train_auroc)
            history['train_precision'].append(train_precision)
            history['train_recall'].append(train_recall)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['test_auroc'].append(test_auroc)
            history['test_precision'].append(test_precision)
            history['test_recall'].append(test_recall)

            if test_auroc > best_test_auroc:
                best_test_auroc = test_auroc
                if save_artifacts:
                    folder_name = "ablation_pyg" if ablation else "pyg"
                    save_dir = os.path.join("results", folder_name)
                    os.makedirs(save_dir, exist_ok=True)

                    save_path = os.path.join(save_dir, "pyg_model.pt")
                    torch.save(model.state_dict(), save_path)

                best_metrics = {
                    'loss': test_loss,
                    'acc': test_acc,
                    'auroc': test_auroc,
                    'precision': test_precision,
                    'recall': test_recall
                }

            print(
                f"epoch {epoch+1:>3,}: "
                f"train loss: {train_loss:.4f}, train acc: {train_acc:.4%}, train AUROC: {train_auroc:.4f}, "
                f"test loss: {test_loss:.4f}, test acc: {test_acc:.4%}, test AUROC: {test_auroc:.4f}"
            )

        if save_artifacts:
            print(f"--> Saved new best model (Auroc: {best_test_auroc:.4%})")
        
            # 1. Loss Plot
            save_plot(
                ablation=ablation,
                train_metric=history['train_loss'],
                test_metric=history['test_loss'],
                metric_name="Loss"
            )

            # 2. AUROC Plot
            save_plot(
                ablation=ablation,
                train_metric=history['train_auroc'],
                test_metric=history['test_auroc'],
                metric_name="AUROC"
            )

            # 3. Accuracy Plot
            save_plot(
                ablation=ablation,
                train_metric=history['train_acc'],
                test_metric=history['test_acc'],
                metric_name="Accuracy"
            )

            # 4. Precision Plot
            save_plot(
                ablation=ablation,
                train_metric=history['train_precision'],
                test_metric=history['test_precision'],
                metric_name="Precision"
            )

            # 5. Recall Plot
            save_plot(
                ablation=ablation,
                train_metric=history['train_recall'],
                test_metric=history['test_recall'],
                metric_name="Recall"
            )

            print(f"\nCharts saved to 'charts' directory.")

        return best_metrics, history

    def train_step(self, model, epoch):
        model.train()
        running_train_loss = 0.0

        acc_metric = BinaryAccuracy().to(self.device)
        auroc_metric = BinaryAUROC().to(self.device)
        precision_metric = BinaryPrecision().to(self.device)
        recall_metric = BinaryRecall().to(self.device)
        probs = []
        targets = []

        with tqdm(self.train_loader, leave=False, desc=f"epoch {epoch+1:>3,}") as t_dataloader:
            for data in t_dataloader:
                x, y = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                output = model(x.x, x.edge_index, x.batch)
                loss = self.criterion(output, y)
                running_train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                probs.append(torch.softmax(output.detach(), dim=1)[:, 1])
                targets.append(y)

        probs = torch.cat(probs)
        targets = torch.cat(targets)
        train_loss = running_train_loss / len(self.train_loader)
        train_acc = acc_metric(probs, targets).item()
        train_auroc = auroc_metric(probs, targets).item()
        train_precision = precision_metric(probs, targets).item()
        train_recall = recall_metric(probs, targets).item()
        return train_loss, train_acc, train_auroc, train_precision, train_recall

    def testing_step(self, model):
        model.eval()
        running_test_loss = 0.0

        acc_metric = BinaryAccuracy().to(self.device)
        auroc_metric = BinaryAUROC().to(self.device)
        precision_metric = BinaryPrecision().to(self.device)
        recall_metric = BinaryRecall().to(self.device)
        probs = []
        targets = []

        with torch.no_grad():
            with tqdm(self.test_loader, leave=False) as t_dataloader:
                for data in t_dataloader:
                    x, y = data[0].to(self.device), data[1].to(self.device)
                    output = model(x.x, x.edge_index, x.batch)
                    loss = self.criterion(output, y)
                    running_test_loss += loss.item()

                    probs.append(torch.softmax(output, dim=1)[:, 1])
                    targets.append(y.float())

        probs = torch.cat(probs)
        targets = torch.cat(targets)
        test_loss = running_test_loss / len(self.test_loader)
        test_acc = acc_metric(probs, targets).item()
        test_auroc = auroc_metric(probs, targets).item()
        test_precision = precision_metric(probs, targets).item()
        test_recall = recall_metric(probs, targets).item()
        return test_loss, test_acc, test_auroc, test_precision, test_recall
