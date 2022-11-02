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


def main():
    model = SportsGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    # TODO: get dataloader
    
    for epoch in range(epochs):
        train(model, train_dataloader, loss_fn, optimizer)


if __name__ == "__main__":
    main()