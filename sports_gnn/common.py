import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class GATBlock(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, edge_dim, num_layers=2, heads=1):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(nn.BatchNorm1d(dim_in))
        self.convs.append(pyg_nn.GATConv(dim_in, dim_h, heads, dropout=0.5, edge_dim=edge_dim))
        for _ in range(self.num_layers-2):
            self.convs.append(nn.BatchNorm1d(dim_h*heads))
            self.convs.append(pyg_nn.GATConv(dim_h*heads, dim_h, heads, dropout=0.5, edge_dim=edge_dim))
        self.convs.append(nn.BatchNorm1d(dim_h*heads))
        self.convs.append(pyg_nn.GATConv(dim_h*heads, dim_out, heads, dropout=0.5, edge_dim=edge_dim))
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, m in enumerate(self.convs):
            if isinstance(m, (pyg_nn.GATConv, pyg_nn.GATv2Conv)):
                x = m(x, edge_index, edge_attr)
            elif isinstance(m, nn.BatchNorm1d):
                x = m(x)
            if i != len(self.convs) - 1:
                x = F.leaky_relu(x, 0.1)
                x = F.dropout(x, p=0.5)
        return F.leaky_relu(x, 0.1), edge_index

class GATv2Block(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, edge_dim, num_layers=2, heads=1):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(nn.BatchNorm1d(dim_in))
        self.convs.append(pyg_nn.GATv2Conv(dim_in, dim_h, heads, dropout=0.5, edge_dim=edge_dim))
        for _ in range(self.num_layers-2):
            self.convs.append(nn.BatchNorm1d(dim_h*heads))
            self.convs.append(pyg_nn.GATv2Conv(dim_h*heads, dim_h, heads, dropout=0.5, edge_dim=edge_dim))
        self.convs.append(nn.BatchNorm1d(dim_h*heads))
        self.convs.append(pyg_nn.GATv2Conv(dim_h*heads, dim_out, heads, dropout=0.5, edge_dim=edge_dim))
        
    def forward(self, graph_data, x):
        edge_index, edge_attr = graph_data.edge_index, graph_data.edge_attr
        for i, m in enumerate(self.convs):
            if isinstance(m, (pyg_nn.GATConv, pyg_nn.GATv2Conv)):
                x = m(x, edge_index, edge_attr)
            elif isinstance(m, nn.BatchNorm1d):
                x = m(x)
            if i != len(self.convs) - 1:
                x = F.leaky_relu(x, 0.1)
                x = F.dropout(x, p=0.5)
        return F.leaky_relu(x, 0.1), edge_index
    

class GCNBlock(nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(nn.BatchNorm1d(dim_in))
        self.convs.append(pyg_nn.GCNConv(dim_in, dim_h))
        for _ in range(self.num_layers-2):
            self.convs.append(nn.BatchNorm1d(dim_h))
            self.convs.append(pyg_nn.GCNConv(dim_h, dim_h))
        self.convs.append(nn.BatchNorm1d(dim_h))
        self.convs.append(pyg_nn.GCNConv(dim_h, dim_out))
        
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_weight = edge_attr[:, 0]
        for i, m in enumerate(self.convs):
            if isinstance(m, pyg_nn.GCNConv):
                x = m(x, edge_index, edge_weight)
            elif isinstance(m, nn.BatchNorm1d):
                x = m(x)
            if i != len(self.convs) - 1:
                x = F.leaky_relu(x, 0.1)
                x = F.dropout(x, p=0.5)
        return F.leaky_relu(x, 0.1), edge_index    


class SumPool(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_size)
        self.linear2 = nn.Linear(hidden_size, in_features)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.1)
        x = torch.sum(x, dim=0)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.1)
        return x
    
    
class ResLinearBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, in_features)
        
    def forward(self, x):
        x1 = F.dropout(x, p=0.5)
        x1 = self.linear(x1)
        x1 = F.leaky_relu(x + x1, 0.1)
        return x1
    
    
class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        x = F.dropout(x, p=0.5)
        x = self.linear(x)
        x = F.leaky_relu(x, 0.1)
        return x
    
    
class MetaDataEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = LinearBlock(out_features, out_features)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.1)
        x = self.fc2(x)
        return x
    

class LSTMBlock(nn.Module):
    def __init__(self, in_features, hidden_size, num_classes, num_layers):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        # self.batch_norm = nn.BatchNorm1d(in_features)
        self.lstm = nn.LSTM(in_features, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, hn, cn):
        # x = self.batch_norm(x)
        out, (hn, cn) = self.lstm(x, (hn, cn))
        out = F.dropout(out, p=0.5)
        final_out = self.fc(out)
        return final_out, hn, cn
    
    def init(self, device=None):
        h0 = torch.zeros(self.num_layers, self.hidden_size)
        c0 = torch.zeros(self.num_layers, self.hidden_size)
        if device:
            h0, c0 = h0.to(device), c0.to(device)
        return h0, c0
    
    