import torch.nn as nn
import torch.nn.functional as F
from sports_gnn.common import *

class SportsGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_transformers1 = nn.Linear(3, 8)
        self.feature_transformers2 = nn.Linear(8, 8)
        self.conv_block = GATv2Block(dim_in=8, dim_h=8, dim_out=8, edge_dim=2, num_layers=2, heads=2)
        # self.conv_block = GCNBlock(dim_in=3, dim_h=24, dim_out=24, num_layers=2)
        self.sum_pool = SumPool(in_features=16, hidden_size=64)
        self.meta_data_encoder = MetaDataEncoder(in_features=6, out_features=8)
        self.fc1 = LinearBlock(in_features=24, out_features=24)
        self.fc2 = LinearBlock(in_features=24, out_features=8)
        self.fc3 = nn.Linear(in_features=8, out_features=3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, graph_data, meta_data):
        x = self.feature_transformers1(graph_data.x)
        x = F.leaky_relu(x, 0.1)
        x = self.feature_transformers2(x)
        x = F.leaky_relu(x, 0.1)
        
        x, _ = self.conv_block(graph_data, x)
        x = self.sum_pool(x)
        
        meta_data = self.meta_data_encoder(meta_data)
        
        x = torch.cat((x, meta_data), dim=0)
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        x = F.dropout(x, p=0.5)
        x = self.fc3(x)
        
        x = torch.unsqueeze(x, dim=0)
        x = self.softmax(x)
        return x
    
