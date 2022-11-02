import torch
from torch_geometric.data import Data

num_player = 22
num_feature = 3

x = torch.rand(num_player, num_feature)

edge_index = []

for i in range(num_player):
    for j in range(num_player):
        edge_index.append([i, j])

edge_index = torch.tensor(edge_index)

data = Data(x=x, edge_index=edge_index.t().contiguous())