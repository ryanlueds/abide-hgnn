import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, global_mean_pool, BatchNorm

class HGNN_Graph(nn.Module):
    """
    An HGNN model for BINARY graph classification.
    Uses HypergraphConv layers and global mean pooling.
    """
    def __init__(self, in_channels, hidden_channels, dropout=0.5):
        """
        Initializes the model layers.
        
        Args:
            in_channels (int): Dimensionality of input node features.
            hidden_channels (int): Dimensionality of the hidden layers.
            dropout (float): Dropout probability.
        """
        super(HGNN_Graph, self).__init__()
        self.dropout = dropout

        # Hypergraph Convolutional Layers
        self.conv1 = HypergraphConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        self.conv2 = HypergraphConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        
        # Final Classifier
        # Takes the pooled graph embedding and maps it to a single logit
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x, hyperedge_index, batch):
        """
        Defines the forward pass of the model.
        
        Args:
            x (torch.Tensor): Node feature matrix [total_nodes_in_batch, in_channels]
            hyperedge_index (torch.Tensor): Hypergraph connectivity [2, total_hyperedge_edges]
            batch (torch.Tensor): Batch vector [total_nodes_in_batch]
        """
        
        # --- Layer 1 ---
        x = self.conv1(x, hyperedge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # --- Layer 2 ---
        x = self.conv2(x, hyperedge_index)
        x = self.bn2(x)
        x = F.relu(x) 
        
        # --- Pooling (Readout) Layer ---
        # Aggregates node features to get a graph-level representation
        x_pooled = global_mean_pool(x, batch)
        
        # --- Final Classification ---
        # Pass graph embedding through the classifier
        out = self.classifier(x_pooled)
        
        # Return raw logits, squeezed to shape [batch_size]
        return out.squeeze(1)