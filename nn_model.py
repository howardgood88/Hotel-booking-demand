import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# model structure
class linearRegression(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(linearRegression, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_feature, n_hidden),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.Linear(n_hidden, n_output),
            nn.Dropout(p=0.3))

    def forward(self, x):
        out = self.linear(x)
        return out