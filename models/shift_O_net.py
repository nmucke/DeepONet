import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class BranchNet(nn.Module):
    def __init__(self, hidden_neurons = [8, 8, 8], output_size=100):
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
'''
class TrunkNet(nn.Module):
    def __init__(self, hidden_neurons=[8, 8, 8],
                 output_size=100):
        super(TrunkNet, self).__init__()

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

class ShiftNet(nn.Module):
    def __init__(self, hidden_neurons=[8, 8, 8],
                 output_size=100):
        super(ShiftNet, self).__init__()

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

class ScaleNet(nn.Module):
    def __init__(self, hidden_neurons=[8, 8, 8],
                 output_size=100):
        super(ShiftNet, self).__init__()

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
'''
class ShiftDeepONet(nn.Module):
    def __init__(self,
                 branch_input_size=100,
                 trunk_input_size=1,
                 trunk_hidden_neurons=[8, 8, 8],
                 branch_hidden_neurons=[8, 8, 8],
                 shift_hidden_neurons=[8, 8, 8],
                 scale_hidden_neurons=[8, 8, 8],
                 bias_hidden_neurons=[8, 8, 8],
                 num_basis_functions=10,
                 ):
        super(ShiftDeepONet, self).__init__()

        self.trunk_input_size = trunk_input_size
        self.trunk_hidden_neurons = [trunk_input_size] + trunk_hidden_neurons

        self.bias_input_size = trunk_input_size
        self.bias_hidden_neurons = [trunk_input_size] + bias_hidden_neurons

        self.branch_input_size = branch_input_size
        self.branch_hidden_neurons = [branch_input_size] + branch_hidden_neurons

        self.shift_input_size = branch_input_size
        self.shift_hidden_neurons = [branch_input_size] + shift_hidden_neurons

        self.scale_input_size = branch_input_size
        self.scale_hidden_neurons = [branch_input_size] + scale_hidden_neurons

        self.num_basis_functions = num_basis_functions
        self.activation = nn.LeakyReLU()

        self.output_scaling_factor = torch.sqrt(torch.tensor(self.num_basis_functions))

        self.trunk_net = NeuralNet(
                hidden_neurons=self.trunk_hidden_neurons,
                output_size=num_basis_functions
        )
        self.trunk_net = TimeDistributed(self.trunk_net)


        self.bias_net = NeuralNet(
                hidden_neurons=self.bias_hidden_neurons,
                output_size=trunk_input_size
        )

        self.branch_net = BranchNet(
                hidden_neurons=self.branch_hidden_neurons,
                output_size=num_basis_functions
        )

        self.shift_net = BranchNet(
                hidden_neurons=self.shift_hidden_neurons,
                output_size=num_basis_functions
        )
        self.scale_net = BranchNet(
                hidden_neurons=self.scale_hidden_neurons,
                output_size=num_basis_functions
        )

    def forward(self, x_branch, x_trunk):
        x_shift = self.shift_net(x_branch)
        x_scale = self.scale_net(x_branch)

        x_bias = self.bias_net(x_trunk)

        x_trunk = x_trunk*x_scale + x_shift
        x_trunk = x_trunk.view(-1, self.num_basis_functions, self.trunk_input_size)

        x_trunk = self.trunk_net(x_trunk)
        x_trunk = torch.diagonal(x_trunk, dim1=-2, dim2=-1)

        x_branch = self.branch_net(x_branch)
        return torch.sum(x_branch*x_trunk, dim=1, keepdim=True)\
               /self.output_scaling_factor + x_bias
