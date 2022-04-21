import pdb

import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from utils.seed_everything import seed_everything
from data_handling.adv_diff_dataset import AdvDiffDataset
from torch.utils.data import DataLoader
from models.vanilla_O_net import DeepONet
from models.shift_O_net import ShiftDeepONet
from tqdm import tqdm

torch.set_default_dtype(torch.float64)

if __name__ == "__main__":

    seed_everything(0)

    train = True
    cuda = True
    if cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'

    data_string = 'data/advection_diffusion/train_data/adv_diff'
    dataset = AdvDiffDataset(data_string, num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    num_basis_functions = 4
    model = DeepONet(
            branch_input_size=128,
            trunk_input_size=1,
            trunk_hidden_neurons=[16, 16],
            branch_hidden_neurons=[16, 16],
            bias_hidden_neurons=[8],
            num_basis_functions=num_basis_functions,
    ).to(device)
    '''
    model = ShiftDeepONet(
            branch_input_size=128,
            trunk_input_size=1,
            trunk_hidden_neurons=[16, 16],
            branch_hidden_neurons=[16, 16],
            shift_hidden_neurons=[8],
            scale_hidden_neurons=[8],
            bias_hidden_neurons=[8],
            num_basis_functions=num_basis_functions,
    ).to(device)
    '''

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    lol = dataloader.dataset[0]

    num_epochs = 1
    if train:
        pbar = tqdm(range(num_epochs), total=num_epochs,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for epoch in pbar:
            for i, data in enumerate(dataloader):
                y, u_old, u_new = data

                y = y.reshape(-1, 1).to(device)
                u_new = u_new.reshape(-1, 1).to(device)
                u_old = u_old.reshape(-1, 128).to(device)

                optimizer.zero_grad()
                u_pred = model(x_branch=u_old, x_trunk=y)
                loss = nn.MSELoss()(u_pred, u_new)
                loss.backward()
                optimizer.step()

            pbar.set_postfix({"Loss": loss.item()})

    model.eval()

    x_vec = np.linspace(-1, 1, 128)
    x_vec = torch.tensor(x_vec).unsqueeze(1).to(device)
    true_sol = np.load(data_string + '_0.npy', allow_pickle=True).item()
    true_sol = true_sol['sol'][:, 0::16]
    init = true_sol[:, 0]
    preds = [init]
    for i in range(31):
        u_old = torch.tensor(preds[-1]).unsqueeze(0).to(device)
        pred = model(x_branch=u_old, x_trunk=x_vec).detach().cpu().numpy()
        preds.append(pred[:, 0])

    x_vec = np.linspace(-1, 1, 128)
    plt.figure()
    plt.plot(x_vec, true_sol[:, 0], label='True', color='tab:blue')
    plt.plot(x_vec, true_sol[:, 15], color='tab:blue')
    plt.plot(x_vec, true_sol[:, 30], color='tab:blue')

    plt.plot(x_vec, preds[0], '--', label='Pred', color='tab:orange')
    plt.plot(x_vec, preds[15], '--', color='tab:orange')
    plt.plot(x_vec, preds[30], '--', color='tab:orange')
    plt.show()

    x_vec = np.linspace(-1, 1, 128)
    #x_vec = np.tile(x_vec, (num_basis_functions,1)).T
    basis = model.trunk_net(torch.tensor(x_vec).unsqueeze(1).to(device))
    basis = basis.detach().cpu().numpy()
    plt.figure()
    for i in range(4):
        base = basis[:, i]
        plt.plot(x_vec, base)
    plt.show()