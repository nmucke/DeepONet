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
from models.GA_O_net import GeneratorONet, CriticONet
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
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    num_basis_functions = 20
    latent_dim = 32
    generator = GeneratorONet(
            branch_input_size=128,
            trunk_input_size=1,
            trunk_hidden_neurons=[32, 32, 32],
            branch_hidden_neurons=[32, 32, 32],
            scale_hidden_neurons=[8],
            shift_hidden_neurons=[8],
            bias_hidden_neurons=[4],
            latent_dim=latent_dim,
            num_basis_functions=num_basis_functions,
    ).to(device)
    generator.train()

    critic = CriticONet(
             critic_input_size=128 + 1 + 1,
             critic_hidden_neurons=[64, 64, 64, 64]
    ).to(device)
    critic.train()

    generator_optimizer = optim.RMSprop(generator.parameters(), lr=1e-4)
    generator_optimizer_pred = optim.Adam(generator.parameters(), lr=1e-4, weight_decay=1e-7)
    critic_optimizer = optim.RMSprop(critic.parameters(), lr=1e-4)


    num_epochs = 100
    if train:
        pbar = tqdm(range(num_epochs), total=num_epochs,
                    bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for epoch in pbar:
            for i, data in enumerate(dataloader):
                y, u_old, u_new = data

                y = y.reshape(-1, 1).to(device)
                u_new = u_new.reshape(-1, 1).to(device)
                u_old = u_old.reshape(-1, 128).to(device)
                x_latent = torch.randn(y.shape[0], latent_dim).to(device)

                for p in critic.parameters():
                    p.data.clamp_(-5, 5)

                # Train the critic
                critic_optimizer.zero_grad()
                generator.eval()
                critic.train()
                u_pred = generator(x_branch=u_old, x_trunk=y, x_latent=x_latent)

                critic_real_input = torch.cat((u_new, u_old, y), dim=1)
                critic_gen_input = torch.cat((u_pred, u_old, y), dim=1)

                critic_loss = -critic(critic_real_input).mean() \
                        + critic(critic_gen_input).mean()
                critic_loss.backward()
                critic_optimizer.step()

                if i % 5 == 0:
                    generator.train()
                    critic.eval()
                    generator_optimizer.zero_grad()

                    u_pred = generator(x_branch=u_old, x_trunk=y, x_latent=x_latent)
                    critic_gen_input = torch.cat((u_pred, u_old, y), dim=1)
                    gen_loss = -critic(critic_gen_input).mean()
                    gen_loss.backward()
                    generator_optimizer.step()


                    generator_optimizer_pred.zero_grad()

                    u_pred = generator(x_branch=u_old, x_trunk=y, x_latent=x_latent)
                    gen_loss_pred = nn.MSELoss()(u_pred, u_new)
                    gen_loss_pred.backward()
                    generator_optimizer_pred.step()

            if np.abs(gen_loss.item()) > 1e8:
                break

            if np.abs(critic_loss.item()) > 1e8:
                break

            pbar.set_postfix({
                "Gen loss": gen_loss.item(),
                "Gen loss pred": gen_loss_pred.item(),
                "Critic loss": critic_loss.item()
            })

    generator.eval()

    true_sols = []
    for i in range(1000):
        true_sol = np.load(data_string+'_'+str(i)+'.npy', allow_pickle=True).item()
        true_sol = true_sol['sol'][:, 0::16]
        true_sols.append(true_sol)
    true_sols = np.array(true_sols)
    true_sols_mean = np.mean(true_sols, axis=0)
    true_sols_std = np.std(true_sols, axis=0)
    plt.figure()

    x_vec = np.linspace(-1, 1, 128)
    plt.figure()
    plt.plot(x_vec, true_sols_mean[:, 0], label='True', color='tab:blue')
    plt.plot(x_vec, true_sols_mean[:, 15], color='tab:blue')
    plt.plot(x_vec, true_sols_mean[:, 30], color='tab:blue')

    plt.fill_between(x_vec, true_sols_mean[:, 0] - true_sols_std[:, 0],
                     true_sols_mean[:, 0] + true_sols_std[:, 0],
                     alpha=0.1, color='tab:blue')
    plt.fill_between(x_vec, true_sols_mean[:, 15] - true_sols_std[:, 15],
                     true_sols_mean[:, 15] + true_sols_std[:, 15],
                     alpha=0.1, color='tab:blue')
    plt.fill_between(x_vec, true_sols_mean[:, 30] - true_sols_std[:, 30],
                     true_sols_mean[:, 30] + true_sols_std[:, 30],
                     alpha=0.1, color='tab:blue')
    plt.show()


    x_vec = np.linspace(-1, 1, 128)
    x_vec = torch.tensor(x_vec).unsqueeze(1).to(device)
    true_sol = np.load(data_string + '_0.npy', allow_pickle=True).item()
    true_sol = true_sol['sol'][:, 0::16]
    init = true_sol[:, 0]
    pred_mean = [init]
    pred_std = [np.zeros(init.shape)]
    for i in range(31):
        u_old = torch.tensor(pred_mean[-1]).unsqueeze(0).to(device)
        u_old = torch.tile(u_old, (128, 1)).to(device)

        pred = []
        for j in range(1000):
            x_latent = torch.randn(128, latent_dim).to(device)
            u_pred = generator(x_branch=u_old, x_trunk=x_vec, x_latent=x_latent).detach().cpu().numpy()
            pred.append(u_pred[:, 0])

        pred = np.array(pred)
        pred_mean.append(pred.mean(axis=0))
        pred_std.append(pred.std(axis=0))

    pred_mean = np.array(pred_mean).transpose()
    pred_std = np.array(pred_std).transpose()

    x_vec = np.linspace(-1, 1, 128)
    plt.figure()
    plt.plot(x_vec, pred_mean[:, 0], label='True', color='tab:blue')
    plt.plot(x_vec, pred_mean[:, 15], color='tab:blue')
    plt.plot(x_vec, pred_mean[:, 30], color='tab:blue')

    plt.fill_between(x_vec, pred_mean[:, 0] - pred_std[:, 0],
                     pred_mean[:, 0] + pred_std[:, 0],
                     alpha=0.1, color='tab:blue')
    plt.fill_between(x_vec, pred_mean[:, 15] - pred_std[:, 15],
                     pred_mean[:, 15] + pred_std[:, 15],
                     alpha=0.1, color='tab:blue')
    plt.fill_between(x_vec, pred_mean[:, 30] - pred_std[:, 30],
                     pred_mean[:, 30] + pred_std[:, 30],
                     alpha=0.1, color='tab:blue')
    plt.show()