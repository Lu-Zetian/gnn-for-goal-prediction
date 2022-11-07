import os
import torch
import torch.nn as nn
from sports_gnn.model import SportsGNN
from utils import *

# Hyper parameters
learning_rate = 1e-4
epochs = 1
batch_size = 32
weight_decay = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
model_dir = "weights"
model_filename = "model.pth"
root = os.path.dirname(os.path.abspath(__file__))


def train(model, data, loss_fn, optimizer):
    model.train()
    # TODO: single training loop
    for i in range(len(data)):
        hn, cn = model.init(device)
        for j in range(len(data[i].gamestates)//20):
            gamestate = data[i].gamestates[j]
            try:
                graph_data, meta_data, label = gamestate.graph, gamestate.metadata, gamestate.label
            except:
                continue
            label = label_converter(label)
            meta_data = meta_data_converter(meta_data)
            
            print(label, meta_data)
            break
            graph_data, meta_data, label = graph_data.to(device), meta_data.to(device), label.to(device)
            optimizer.zero_grad()
            output, hn, cn = model(graph_data, meta_data, hn, cn)
            hn, cn = hn.detach(), cn.detach()
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            print(f"{j}/{len(data[i].gamestates)}", end="\r")
    

def eval(model, test_dataloader):
    model.eval()
    # TODO: calculate accuracy
    accuracy = 0
    hn, cn = model.init(device)
    with torch.no_grad():
        inputs, lebels = next(iter(test_dataloader))
        inputs, lebels = inputs.to(device), lebels.to(device)
        # for loop
    return


def main():
    if not os.path.isdir(os.path.join(root, model_dir)):
        os.mkdir(os.path.join(root, model_dir))
    
    if load_model:
        model = torch.load(os.path.join(root, model_dir, model_filename))
    else:
        model = SportsGNN().to(device)
        
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    data = load_data("sample_data.pkl")
    
    for epoch in range(epochs):
        train(model, data, loss_fn, optimizer)
        
    # torch.save(model, os.path.join(root, model_dir, model_filename))


if __name__ == "__main__":
    main()