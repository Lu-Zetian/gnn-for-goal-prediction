import os
import pickle, time
import torch
import torch.nn as nn
from sports_gnn.model import SportsGNN
from DataParsing.DataParsing import Match,GameState

# Hyper parameters
learning_rate = 1e-4
epochs = 1
batch_size = 32
weight_decay = 1e-5
model_path = ""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DataLocation = os.path.dirname(os.path.abspath(__file__))


def load_data(filename):
    st = time.time()
    sample = os.path.join(DataLocation, filename)
    with (open(sample,'rb')) as openfile:  # read the parsed data
        print('reading')
        while True:
            try:
                data=pickle.load(openfile)
            except EOFError:
                break
        print('loaded')
    et = time.time()
    print('Reading time:\t',et-st) # loading time
    return data


def label_converter(label):
    onehot_label = torch.zeros(3, dtype=torch.float32)
    if label == 1:
        onehot_label[0] = 1
    elif label == 0:
        onehot_label[1] = 1
    elif label == -1:
        onehot_label[2] = 1
    return onehot_label


def meta_data_converter(meta_data):
    print(meta_data)
    return meta_data


def train(model, data, loss_fn, optimizer):
    model.train()
    # TODO: single training loop
    for i in range(len(data)):
        hn, cn = model.init(device)
        for j in range(len(data[i].gamestates)):
            gamestate = data[i].gamestates[j]
            try:
                graph_data, meta_data, label = gamestate.graph, gamestate.metadata, gamestate.label
            except:
                continue
            label = label_converter(label)
            meta_data = meta_data_converter(meta_data)
            # graph_data, meta_data, label = graph_data.to(device), meta_data.to(device), label.to(device)
            # print(graph_data, meta_data, label, sep="\n")
            # print(graph_data.x)
            # print(label, end=" ")
            # print(j)
            break
            # convert label and meta_data
        
    # for batch_idx, (input, label) in train_dataloader:
    #     input, label = input.to(device), label.to(device)
    #     optimizer.zero_grad()
    #     output, hn, cn = model(input, hn, cn)
    #     hn, cn = hn.detach(), cn.detach()
    #     loss = loss_fn(output, label)
    #     loss.backward()
    #     optimizer.step()
    

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
    model = SportsGNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    data = load_data("sample_data.pkl")
    
    for epoch in range(epochs):
        train(model, data, loss_fn, optimizer)


if __name__ == "__main__":
    main()