import os
import torch
import torch.nn as nn
from sports_gnn.model import SportsGNN
from utils import *
import random

# Hyper parameters
learning_rate = 1e-4
epochs = 1
weight_decay = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = True
model_dir = "weights"
model_filename = "model.pth"
# data_filename = "sample_data.pkl"
data_filename = "data_finalized.pickle"
root = os.path.dirname(os.path.abspath(__file__))


def train(model, data, loss_fn, optimizer):
    model.train()
    for i in range(len(data)):
        hn, cn = model.init(device)
        gamestates_copy = data[i].gamestates.copy()
        for j in range(len(gamestates_copy)):
            try:
                gamestate = gamestates_copy[j]
                graph_data, meta_data, label = gamestate.graph, gamestate.metadata, gamestate.label
                label = label_to_onehot(label)
                meta_data = meta_data_to_vector(meta_data)
                graph_data, meta_data, label = graph_data.to(device), meta_data.to(device), label.to(device)
                optimizer.zero_grad()
                output, hn, cn = model(graph_data, meta_data, hn, cn)
                hn, cn = hn.detach(), cn.detach()
                loss = loss_fn(output, label)
                loss.backward()
                optimizer.step()
                print(f"Training, match: {i}, timestamp: {j}/{len(data[i].gamestates)}", end="\r")
            except:
                    print(f"Train Error at match: {i}, timestamp: {j}")
                    data[i].gamestates.remove(gamestates_copy[j])
                    continue
    

def eval(model, data):
    model.eval()
    num_pred = 0
    num_correct = 0
    i = random.randrange(len(data))
    with torch.no_grad():
        hn, cn = model.init(device)
        for j in range(len(data[i].gamestates)):
            try:
                gamestate = data[i].gamestates[j]
                graph_data, meta_data, label = gamestate.graph, gamestate.metadata, gamestate.label
                label = label_to_index(label)
                meta_data = meta_data_to_vector(meta_data)
                graph_data, meta_data = graph_data.to(device), meta_data.to(device)
                output, hn, cn = model(graph_data, meta_data, hn, cn)
                if j == 100:
                    print(output)
                hn, cn = hn.detach(), cn.detach()
                output = torch.argmax(output)
                num_pred += 1
                if output == label:
                    num_correct += 1
                print(f"Evaluating,  match: {i}, timestamp: {j}/{len(data[i].gamestates)},", end="\r")
            except:
                print(f"Evaluate Error at match: {i}, timestamp: {j}")
                continue
    accuracy = (float)(num_correct)/num_pred
    return accuracy


def main():
    if not os.path.isdir(os.path.join(root, model_dir)):
        os.mkdir(os.path.join(root, model_dir))
    
    if load_model:
        model = torch.load(os.path.join(root, model_dir, model_filename))
    else:
        model = SportsGNN().to(device)
        
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    data = load_data(data_filename)
    
    for epoch in range(epochs):
        train(model, data, loss_fn, optimizer)
        accuracy = eval(model, data)
        print(f"epoch: {epoch+1}, accuracy: {accuracy}")
        
    torch.save(model, os.path.join(root, model_dir, model_filename))


if __name__ == "__main__":
    main()