import os
import torch
import torch.nn as nn
from sports_gnn.model import SportsGNN
from utils import *
import random

# Hyper parameters
learning_rate = 1e-4
epochs = 10
weight_decay = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_eval_match = 3
num_eval_data = 3000
load_model = True
model_dir = "weights"
model_filename = "model.pth"
result_dir = "results"
# data_filename = "sample_data.pkl"
data_filename = "data_finalized.pickle"
root = os.path.dirname(os.path.abspath(__file__))


def train(model, data, loss_fn, optimizer):
    model.train()
    data_copy = data.copy()
    num_error = 0
    for i in range(len(data_copy)):
        try:
            graph_data, meta_data, label = data_copy[i].graph, data_copy[i].metadata, data_copy[i].label
            label = label_to_onehot(label)
            meta_data = meta_data_to_vector(meta_data)
            graph_data, meta_data, label = graph_data.to(device), meta_data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(graph_data, meta_data)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            print(f"Training, progress: {i}/{len(data_copy)}", end="\r")
        except:
            # print(f"Train Error at {i}/{len(data_copy)}")
            data.pop(i-num_error)
            num_error += 1
            continue
    return data
    

def eval(model, data):
    model.eval()
    num_pred = 0
    num_correct = 0
    with torch.no_grad():
        for _ in range(num_eval_data):
            try:
                gamestate = random.choice(data)
                graph_data, meta_data, label = gamestate.graph, gamestate.metadata, gamestate.label
                label = label_to_index(label)
                meta_data = meta_data_to_vector(meta_data)
                graph_data, meta_data = graph_data.to(device), meta_data.to(device)
                output = model(graph_data, meta_data)
                output = torch.argmax(output)
                num_pred += 1
                if output == label:
                    num_correct += 1
            except:
                continue
    accuracy = (float)(num_correct)/num_pred
    return accuracy


def main():
    if not os.path.isdir(os.path.join(root, model_dir)):
        os.mkdir(os.path.join(root, model_dir))
    if not os.path.isdir(os.path.join(root, result_dir)):
        os.mkdir(os.path.join(root, result_dir))
    
    if load_model:
        model = torch.load(os.path.join(root, model_dir, model_filename))
    else:
        model = SportsGNN().to(device)
        
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    data = load_data(data_filename)
    
    test_data = []
    for _ in range(num_eval_match):
        index = random.randrange(len(data))
        test_data.append((data.pop(index)).gamestates)
    test_data = list(chain.from_iterable(test_data))
    
    train_data = flatten_data(data)
    
    for epoch in range(epochs):
        random.shuffle(train_data)
        train_data = train(model, train_data, loss_fn, optimizer)
        accuracy = eval(model, test_data)
        print(f"Finished epoch: {epoch+1}, accuracy: {accuracy}")
        
    torch.save(model, os.path.join(root, model_dir, model_filename))


if __name__ == "__main__":
    main()