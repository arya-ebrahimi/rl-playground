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

class ActorCritic(nn.Module):
    def __init__(self, channels, action_space):
        super(ActorCritic, self).__init__()

        self.action_space = action_space
        
        self.conv1 = nn.Conv2d(channels, out_channels=32, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=0)
        
        self.flatten = nn.Flatten()
        
        self.rep = nn.Linear(64*8*8, 512)
        
        self.actor1 = nn.Linear(512, 64)
        self.actor2 = nn.Linear(64, self.action_space)
        
        self.critic1 = nn.Linear(512, 64)
        self.critic2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # print(x)
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        x=x.to(torch.device('cuda'))
        # print(x)
        x = x / 255.0
        
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = self.flatten(x)
        x = x.reshape((-1, 64*8*8))
        
        x = self.rep(x)
        
        policy = F.relu(self.actor1(x))
        policy = self.actor2(policy)
        policy = F.softmax(policy, dim=-1)
        
        value = F.relu(self.critic1(x))
        value = self.critic2(value)
        
        return policy, value
        
