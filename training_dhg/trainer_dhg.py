import torch
import torch.nn as nn
from tqdm import tqdm
import config as config


class Trainer(object):

    def __init__(self, device, optimizer, train_loader, test_loader, criterion=nn.CrossEntropyLoss()):
        self.device = device
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion


    def fit(self, model):
        num_epochs = config.EPOCHS
        for epoch in range(num_epochs):
            train_loss = self.train_step(model, epoch)
            test_loss, test_acc = self.testing_step(model)
            print(f"epoch {epoch+1:>3,}: training loss: {train_loss:.4f}, test loss: {test_loss:.4f}, test accuracy: {test_acc:.4%}")
    
    
    def train_step(self, model, epoch):
        model.train()
        running_train_loss = 0.0

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

                batch_loss = batch_loss / batch_size
                batch_loss.backward()
                self.optimizer.step()
                running_train_loss += batch_loss.item() * batch_size

        train_loss = running_train_loss / len(self.train_loader.dataset)
        return train_loss

    def testing_step(self, model):
        model.eval()
        running_test_loss = 0.0
        running_accuracy = 0
        total = 0

        with torch.no_grad():
            for xs, ys, hgs in self.test_loader:
                for x, y, hg in zip(xs, ys, hgs):
                    x, y, hg = x.to(self.device), y.to(self.device).unsqueeze(0), hg.to(self.device)
                    output = model(x, hg).mean(dim=0, keepdim=True)
                    loss = self.criterion(output, y)
                    running_test_loss += loss.item()
                    preds = torch.argmax(output, dim=1)
                    running_accuracy += torch.sum(preds == y)
                    total += y.size(0)

        test_loss = running_test_loss / len(self.test_loader.dataset)
        test_acc = running_accuracy / total
        return test_loss, test_acc
