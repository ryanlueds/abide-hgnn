import torch
import torch.nn as nn
from tqdm import tqdm
import config as config
from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryPrecision, BinaryRecall
from plotter import save_plot

class Trainer(object):

    def __init__(self, device, optimizer, train_loader, test_loader, criterion=nn.CrossEntropyLoss()):
        self.device = device
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion


    def fit(self, model):
        history = {
            'train_loss': [], 'train_acc': [], 'train_auroc': [], 'train_precision': [], 'train_recall': [],
            'test_loss': [], 'test_acc': [], 'test_auroc': [], 'test_precision': [], 'test_recall': []
        }

        num_epochs = config.EPOCHS
        best_test_acc = 0.0
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

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), "dhg_model.pt")
                
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

        print(f"--> Saved new best model (Acc: {best_test_acc:.4%})")

        save_plot(history['train_loss'], history['test_loss'], "Loss")
        save_plot(history['train_auroc'], history['test_auroc'], "AUROC")
        save_plot(history['train_acc'], history['test_acc'], "Accuracy")
        save_plot(history['train_precision'], history['test_precision'], "Precision")
        save_plot(history['train_recall'], history['test_recall'], "Recall")
        print(f"\nCharts saved to 'charts' directory.")

        return best_metrics

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
            for xs, ys, hgs in t_dataloader:
                self.optimizer.zero_grad()
                batch_loss = 0.0
                batch_size = len(ys)
                for x, y, hg in zip(xs, ys, hgs):
                    x, y, hg = x.to(self.device), y.to(self.device).unsqueeze(0), hg.to(self.device)
                    output = model(x, hg).mean(dim=0, keepdim=True)
                    loss_i = self.criterion(output, y)
                    batch_loss += loss_i
                    probs.append(torch.softmax(output, dim=1)[:, 1])
                    targets.append(y)

                batch_loss = batch_loss / batch_size
                batch_loss.backward()
                self.optimizer.step()
                running_train_loss += batch_loss.item() * batch_size

        probs = torch.cat(probs)
        targets = torch.cat(targets)
        train_loss = running_train_loss / len(self.train_loader.dataset)
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
            for xs, ys, hgs in self.test_loader:
                for x, y, hg in zip(xs, ys, hgs):
                    x, y, hg = x.to(self.device), y.to(self.device).unsqueeze(0), hg.to(self.device)
                    output = model(x, hg).mean(dim=0, keepdim=True)
                    loss = self.criterion(output, y)
                    running_test_loss += loss.item()
                    probs.append(torch.softmax(output, dim=1)[:, 1])
                    targets.append(y.float())

        probs = torch.cat(probs)
        targets = torch.cat(targets)
        test_loss = running_test_loss / len(self.test_loader.dataset)
        test_acc = acc_metric(probs, targets).item()
        test_auroc = auroc_metric(probs, targets).item()
        test_precision = precision_metric(probs, targets).item()
        test_recall = recall_metric(probs, targets).item()
        return test_loss, test_acc, test_auroc, test_precision, test_recall