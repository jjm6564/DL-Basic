import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

#model = nn.Linear(1,1)

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1,1)
    def forward(self,x):
        return self.linear(x)
model = LinearRegressionModel()

class MultiVariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)

    def forward(self,x):
        return self.linear(x)
model2 = MultiVariateLinearRegressionModel()