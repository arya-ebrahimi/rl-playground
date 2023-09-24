import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLP, self).__init__()
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        x=x.to(torch.device('cuda'))

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        o = self.layer3(x)
        
        return o