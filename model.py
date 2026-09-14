import torch
import torch.nn as nn


class AircraftClassifier(nn.Module):
    def __init__(self, in_channels, filters=32, kernel_size=3, pool_size=2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size)
        self.pool1 = nn.MaxPool2d(pool_size)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size)
        self.pool2 = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.LazyLinear(128)
        self.drouput2 = nn.Dropout(0.5)
        self.fc2 = nn.LazyLinear(1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.drouput2(x)
        x = self.fc2(x)
        return x
