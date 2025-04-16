import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

# Constants from your code
num_atom_type = 119
num_chirality_tag = 4
num_bond_type = 4  # Updated to match bond_to_feature_vector output (3D vectors)

class GINEConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim):
        super(GINEConvLayer, self).__init__()
        # MLP for GINEConv
        nn_layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.conv = GINEConv(nn=nn_layer, eps=0, train_eps=True)
        # Transform edge features to match node feature dimension
        self.edge_transform = nn.Linear(edge_dim, in_channels)

    def forward(self, x, edge_index, edge_attr):
        # Convert edge_attr to float32 to match model weights
        edge_attr = edge_attr.float()  # From int64 to float32
        edge_attr = self.edge_transform(edge_attr)  # (num_edges, in_channels)
        return self.conv(x, edge_index, edge_attr)

class GCNNet(nn.Module):
    def __init__(self, n_output=2, emb_dim=78, num_features_xd=78, dropout=0.2):
        super(GCNNet, self).__init__()

        # Embeddings for node features
        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        # Activation and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # GINEConv layers
        self.drug1_conv1 = GINEConvLayer(num_features_xd, num_features_xd, edge_dim=num_bond_type)
        self.drug1_conv2 = GINEConvLayer(num_features_xd, num_features_xd * 3, edge_dim=num_bond_type)
        self.drug1_conv3 = GINEConvLayer(num_features_xd * 3, 128, edge_dim=num_bond_type)

        # Fully connected layers
        self.fc_g1 = nn.Linear(128, 1024)
        self.fc_g2 = nn.Linear(1024, 128)
        self.line = nn.Linear(312, 128)

        self.n_output = n_output

    def forward(self, data1):
        # Extract inputs
        x1, edge_index1, edge_attr1, batch1 = data1.x, data1.edge_index, data1.edge_attr, data1.batch
        
        # Node embedding
        x1 = self.x_embedding1(x1[:, 0].int()) + self.x_embedding2(x1[:, 1].int())  # (num_nodes, emb_dim)

        # GINEConv layers
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