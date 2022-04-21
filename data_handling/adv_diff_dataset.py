import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset


class AdvDiffDataset(Dataset):
    def __init__(self, data_string=None, num_samples=None):
        self.data_string = data_string
        self.num_samples = num_samples

        self.x_sensors = np.linspace(-1, 1, 128)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        data = np.load(f'{self.data_string}_{idx}.npy', allow_pickle=True).item()
        data = data['sol'][:, 0::16]

        u_old = data[:, 0:-1]
        u_new = data[:, 1:]
        y = np.repeat(self.x_sensors, (u_old.shape[-1],))

        u_old = np.tile(u_old, (1, len(self.x_sensors))).transpose()
        u_new = u_new.flatten()

        return y, u_old, u_new