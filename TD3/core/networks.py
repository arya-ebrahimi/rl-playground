import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    
    def __init__(self, dims, device):
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dims = dims
        self.device = device
        
        for i in range(len(self.dims)-1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
        
    def forward(self, x, actions):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        x = x.to(torch.device(self.device))
        
        x = torch.cat((x, actions), dim=1)
        
        for i in range(len(self.dims)-2):
            x = F.relu(self.layers[i](x))

        o = self.layers[len(self.dims)-2](x)
            
        return o
    
class Actor(nn.Module):
    
    def __init__(self, dims, max_action, device):
        super(Actor, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dims = dims
        self.max_action = max_action
        self.device = device
        
        for i in range(len(self.dims)-1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
        
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        x = x.to(torch.device(self.device))
        
        for i in range(len(self.dims)-2):
            x = F.relu(self.layers[i](x))

        o = self.layers[len(self.dims)-2](x)
        o = self.max_action * torch.tanh(o)
        
        return o
    
class Critic(nn.Module):
    def __init__(self, dims, device):
        super(Critic, self).__init__()
        
        self.dims = dims
        self.device = device
        
        self.q1 = MLP(dims=dims, device=device)
        self.q2 = MLP(dims=dims, device=device)
    
    def forward(self, x, actions):
        
        q1 = self.q1(x, actions)
        q2 = self.q2(x, actions)
        
        return q1, q2
    
    def get_q1(self, x, actions):
        return self.q1(x, actions)