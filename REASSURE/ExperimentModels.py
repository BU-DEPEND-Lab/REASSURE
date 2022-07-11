import torch, torchvision, numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, size):
        super(MLP, self).__init__()
        self.size = size
        self.layers = nn.ModuleList()
        for i in range(len(size)-1):
            self.layers.append(nn.Linear(size[i], size[i+1]))

    def forward(self, x):
        x = x.view(-1, self.size[0])
        for layer in self.layers:
            if layer is self.layers[-1]:
                x = layer(x)
            else:
                x = F.relu(layer(x))
        return x

    def allHiddenNeurons(self, x):
        hidden_neurons = []
        x = x.view(self.size[0])
        for layer in self.layers[:-1]:
            if layer is self.layers[0]:
                x = layer(x)
            else:
                x = layer(F.relu(x))
            hidden_neurons.append(x)
        return torch.cat(hidden_neurons, dim=-1)

    def activationPatterns(self, x):
        x_activation_pattern = self.all_hidden_neurons(x) > 0
        return [entry.item() for entry in x_activation_pattern]







