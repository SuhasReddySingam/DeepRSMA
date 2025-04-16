import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

# Constants from your code
num_atom_type = 119
num_chirality_tag = 4
num_bond_type = 5  # Matches your dataset (assuming bond_to_feature_vector outputs 5-dim vectors)
num_bond_stereo = 6
num_bond_direction = 3

class GINEConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(GINEConvLayer, self).__init__()
        # Define the MLP for GINEConv
        nn_layer = nn.Sequential(
            nn.Linear(in_channels + edge_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.conv = GINEConv(nn=nn_layer, eps=0, train_eps=True)  # Learnable epsilon

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, edge_attr)

class GCNNet(nn.Module):
    def __init__(self, n_output=2, emb_dim=78, num_features_xd=78, dropout=0.2):
        super(GCNNet, self).__init__()

        # Embeddings for node features (atom type and chirality)
        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # GINEConv layers replacing GCNConv
        self.drug1_conv1 = GINEConvLayer(num_features_xd, num_features_xd, edge_dim=num_bond_type)
        self.drug1_conv2 = GINEConvLayer(num_features_xd, num_features_xd * 3, edge_dim=num_bond_type)
        self.drug1_conv3 = GINEConvLayer(num_features_xd * 3, 128, edge_dim=num_bond_type)

        # Fully connected layers (unchanged)
        self.fc_g1 = nn.Linear(128, 1024)
        self.fc_g2 = nn.Linear(1024, 128)
        self.line = nn.Linear(312, 128)  # Note: Unused in current forward pass

        self.n_output = n_output
        self.dropout = nn.Dropout(dropout)  # Note: Two dropout definitions; using one

    def forward(self, data1):
        # Extract node features, edge indices, edge attributes, and batch
        x1, edge_index1, edge_attr1, batch1 = data1.x, data1.edge_index, data1.edge_attr, data1.batch
        
        # Initial node embedding from atom type and chirality
        x1 = self.x_embedding1(x1[:, 0].int()) + self.x_embedding2(x1[:, 1].int())  # (num_nodes, emb_dim)

        # GINEConv layers with edge features
        x1 = self.drug1_conv1(x1, edge_index1, edge_attr1)  # (num_nodes, num_features_xd)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)

        x1 = self.drug1_conv2(x1, edge_index1, edge_attr1)  # (num_nodes, num_features_xd * 3)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)

        x1 = self.drug1_conv3(x1, edge_index1, edge_attr1)  # (num_nodes, 128)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)

        # Graph-level representation
        emb = global_mean_pool(x1, batch1)  # (batch_size, 128)

        return x1, emb

# Example usage
# data1 = Data(x=torch.tensor([[6, 0], [6, 0]]),  # 2 carbon atoms
#              edge_index=torch.tensor([[0, 1], [1, 0]]).t(),
#              edge_attr=torch.tensor([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]),  # Single bonds
#              batch=torch.tensor([0, 0]))
# model = GCNNet()
# node_emb, graph_emb = model(data1)