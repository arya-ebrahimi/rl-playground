import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    
    def __init__(self, dims, mode):
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dims = dims
        self.mode = mode
        
        for i in range(len(self.dims)-1):
            if i == 1 and self.mode=='critic':
                self.layers.append(nn.Linear(self.dims[i], self.dims[i+1]-1))
            else:
                self.layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            
        
    def forward(self, x, actions=None):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        x=x.to(torch.device('cuda'))
        for i in range(len(self.dims)-2):
            if actions != None and i == 2:
                x = torch.cat((x, actions), dim=1)
            x = F.relu(self.layers[i](x))

        o = self.layers[len(self.dims)-2](x)
        if self.mode == 'actor':
            o = torch.tanh(o)
            
        return o