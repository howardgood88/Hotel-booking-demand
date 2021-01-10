import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# adr model structure
class linearRegression(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(linearRegression, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_feature, 700),
            nn.ReLU(),
            nn.Linear(700, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(500, 100),
            nn.Linear(100, n_output)
        )
            
    def forward(self, x):
        out = self.linear(x)
        return out


# is_cancel model structure
class BinaryClassifier(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(BinaryClassifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_feature, 500),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.Dropout(p=0.3),
            nn.Linear(100, n_output),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.linear(x)
        return out


# label model structure
class TenClassClassifier(torch.nn.Module):
    def __init__(self, n_feature, n_output):
        super(TenClassClassifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_feature, 10),
            nn.Tanh(),
            # nn.Dropout(p=0.3),
            nn.Linear(10, n_output),
            # nn.Softmax()
        )

    def forward(self, x):
        out = self.linear(x)
        return out