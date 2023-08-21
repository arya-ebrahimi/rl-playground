import torch
import gymnasium as gym
from fta import FTA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False,
                            'axes.edgecolor':'black'})

activation = FTA(tiles=4, bound_low=0, bound_high=1, eta=0.0, input_dim=1)


fig, axs = plt.subplots(4)

l = []
for i in range (0, 101):
    x = torch.tensor(i/100)
    l.append(activation(x).squeeze().numpy())

l = np.array(l)

for i in range(l.shape[1]):
    
    axs[i].plot(np.linspace(0, 1, l.shape[0]), l[:,i], drawstyle='steps-pre', linewidth=2)
    if i < l.shape[1]-1:
        # frame = axs[i].gca()
        axs[i].axes.get_xaxis().set_ticks([])
    axs[i].set(ylabel='Bin '+str(i+1))

# plt.plot(np.linspace(0, 1, l.shape[0]), l[:,0])

plt.show()

# print(activation(x))