import torch
import torch.nn as nn
from sports_gnn.model import SportsGNN

# Hyper parameters
learning_rate = 1e-4
epochs = 10
batch_size = 32
weight_decay = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_dataloader, loss_fn, optimizer):
    model.train()
    # TODO: single training loop
    hn, cn = model.init(device)
    for batch_idx, (input, label) in train_dataloader:
        input, label = input.to(device), label.to(device)
        optimizer.zero_grad()
        output, hn, cn = model(input, hn, cn)
        hn, cn = hn.detach(), cn.detach()
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
    

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
    
    # TODO: get dataloader
    
    for epoch in range(epochs):
        train(model, train_dataloader, loss_fn, optimizer)


if __name__ == "__main__":
    main()