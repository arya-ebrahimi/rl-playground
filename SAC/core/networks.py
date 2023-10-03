import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from torch.nn.functional import softplus

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
        
        for i in range(len(self.dims)-2):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            
        self.mu = torch.nn.Linear(self.dims[-2], self.dims[-1])
        self.logstd = torch.nn.Linear(self.dims[-2], self.dims[-1])
        
    def forward(self, x, get_logprob=False):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        x = x.to(torch.device(self.device))
        
        for i in range(len(self.dims)-2):
            x = F.relu(self.layers[i](x))

        
        mu = self.mu(x)
    
        logstd = self.logstd(x)
        logstd = torch.clamp(logstd, min=-20, max=2)   

        std = logstd.exp()
        dist = Normal(mu, std)
        
        z = dist.rsample()
        action = torch.tanh(z)
        
        if get_logprob:
            logprob = dist.log_prob(z).sum(axis=1, keepdim=True) - (2 * (np.log(2) - z - F.softplus(-2 * z))).sum(axis=1, keepdim=True)
        else:
            logprob = None
        
        mean = torch.tanh(mu) 
        
        return action, logprob, mean
    
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
