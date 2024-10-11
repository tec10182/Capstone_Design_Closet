import torch.nn as nn
import torch.nn.init as init


class Embedder(nn.Module):
    def __init__(self, dim=64):
        super(Embedder, self).__init__()
        self.fc1 = nn.Linear(1000, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.output(x)
        return x
