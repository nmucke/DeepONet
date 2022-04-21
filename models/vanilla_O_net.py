import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch

class BranchNet(nn.Module):
    def __init__(self, input_size=100, hidden_neurons = [8, 8, 8], output_size=100):
        super(BranchNet, self).__init__()

        self.hidden_neurons = hidden_neurons
        self.activation = nn.LeakyReLU()

        self.input_layer = nn.Linear(self.hidden_neurons[0], self.hidden_neurons[1])

        self.hidden_layers = nn.ModuleList()
        for i in range(1, len(hidden_neurons)-1):
            self.hidden_layers.append(
                    nn.Linear(hidden_neurons[i],hidden_neurons[i+1])
            )
        self.output_layer = nn.Linear(hidden_neurons[-1], output_size, bias=True)

    def forward(self, x):

        x = self.activation(self.input_layer(x))
        x_in = torch.clone(x)
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x += x_in
        return self.output_layer(x)

class NeuralNet(nn.Module):
    def __init__(self, hidden_neurons=[8, 8, 8],
                 output_size=100):
        super(NeuralNet, self).__init__()

        self.hidden_neurons = hidden_neurons
        self.output_size = output_size
        self.activation = nn.LeakyReLU()

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_neurons) - 1):
            self.hidden_layers.append(
                    nn.Linear(hidden_neurons[i], hidden_neurons[i + 1])
            )
        self.output_layer = nn.Linear(hidden_neurons[-1], output_size,
                                      bias=True)

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        return self.output_layer(x)

class DeepONet(nn.Module):
    def __init__(self,
                 branch_input_size=100,
                 trunk_input_size=1,
                 trunk_hidden_neurons=[8, 8, 8],
                 branch_hidden_neurons=[8, 8, 8],
                 bias_hidden_neurons=[8, 8, 8],
                 num_basis_functions=10,
                 ):
        super(DeepONet, self).__init__()

        self.trunk_input_size = trunk_input_size
        self.trunk_hidden_neurons = [trunk_input_size] + trunk_hidden_neurons

        self.bias_input_size = trunk_input_size
        self.bias_hidden_neurons = [trunk_input_size] + bias_hidden_neurons

        self.branch_input_size = branch_input_size
        self.branch_hidden_neurons = [branch_input_size] + branch_hidden_neurons


        self.num_basis_functions = num_basis_functions
        self.activation = nn.LeakyReLU()

        self.output_scaling_factor = torch.sqrt(torch.tensor(self.num_basis_functions))

        self.trunk_net = NeuralNet(
                hidden_neurons=self.trunk_hidden_neurons,
                output_size=num_basis_functions
        )


        self.bias_net = NeuralNet(
                hidden_neurons=self.bias_hidden_neurons,
                output_size=trunk_input_size
        )
        self.branch_net = BranchNet(input_size=self.branch_input_size,
                                    hidden_neurons=self.branch_hidden_neurons,
                                    output_size=num_basis_functions)

    def forward(self, x_branch, x_trunk):
        x_bias = self.bias_net(x_trunk)
        x_trunk = self.trunk_net(x_trunk)
        x_branch = self.branch_net(x_branch)
        return torch.sum(x_branch*x_trunk, dim=1, keepdim=True)\
               /self.output_scaling_factor + x_bias
