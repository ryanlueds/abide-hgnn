import torch.nn as nn
from torch_geometric.nn import HypergraphConv, global_mean_pool, AttentionalAggregation
import torch.nn.functional as F


class HGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes=2, dropout=0.3):
        super().__init__()
        self.conv1 = HypergraphConv(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = HypergraphConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

        self.att_pool = AttentionalAggregation(gate_nn=nn.Sequential(nn.Linear(hidden_dim, 1)))
        self.fc = nn.Linear(hidden_dim, num_classes)


    def forward(self, x, hyperedge_index, batch):
        x = self.conv1(x, hyperedge_index)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.conv2(x, hyperedge_index)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.head(x) + x

        x = self.att_pool(x, batch)
        x = self.fc(x)
        return x
    