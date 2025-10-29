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
            train_loss = self.train_step(model)
            test_loss, test_acc = self.testing_step(model)
            print(f"epoch {epoch+1:>4,}: training loss: {train_loss:.4f}, test loss: {test_loss:.4f}, test accuracy: {test_acc:.4%}")
        

    def train_step(self, model):
        model.train()
        running_train_loss = 0.0

        with tqdm(self.train_loader, leave=False) as t_dataloader: 
            for data in t_dataloader:
                x, y = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                output = model(x.x, x.edge_index, x.batch)
                loss = self.criterion(output, y)
                running_train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

        train_loss = running_train_loss / len(self.train_loader)
        return train_loss


    def testing_step(self, model):
        model.eval()
        running_test_loss = 0.0
        running_accuracy = 0.0
        total = 0

        with torch.no_grad():
            with tqdm(self.test_loader, leave=False) as t_dataloader: 
                for data in t_dataloader:
                    x, y = data[0].to(self.device), data[1].to(self.device)
                    output = model(x.x, x.edge_index, x.batch)
                    preds = torch.argmax(output, dim=1)
                    loss = self.criterion(output, y)
                    running_test_loss += loss.item()
                    running_accuracy += torch.sum(preds == y)
                    total += y.numel()
        
        test_loss = running_test_loss / len(self.test_loader)
        test_acc = running_accuracy / total
        return test_loss, test_acc
