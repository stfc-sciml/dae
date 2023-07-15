#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 2023
"""
import torch

import os
import numpy as np
import matplotlib.pyplot as plt

def draw_recover(save_path, model, dataloader, device, epoch):
    model.eval()

    h = w = 84

    for tidx, data in enumerate(dataloader):
        z = model.latent(data['x'].to(device))
        if tidx == 0:
            break

    z = z.detach().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize = (10, 4))
    p = 0
    for i in range(3):
      for j in range(i+1, 3):
        axs[p].scatter(z[:,i],z[:,j], rasterized = True)
        axs[p].set_title('corr between {} and {}'.format(i, j))
        p+=1
    plt.tight_layout()
    save_name = os.path.join(save_path, 'Latent_{}.png'.format(epoch))
    fig.savefig(save_name)
    plt.close()

    z_mean = z.mean(0, keepdims=True)
    z_min = np.percentile(z,1, axis = 0)
    z_max = np.percentile(z,99, axis = 0)

    fig, axs = plt.subplots(1, 3, figsize = (10, 4))
    p = 0
    for i in range(3):
      for j in range(i+1, 3):

        nx = ny = 10
        x_values = np.linspace(z_min[i], z_max[i], nx)
        y_values = np.linspace(z_min[j], z_max[j], ny)

        canvas = np.empty(((h+2)*ny, (w+2)*nx))

        for yv in range(len(y_values)):
            for xv in range(len(x_values)):
                z_mu = z_mean.copy()
                z_mu[0, i] = x_values[xv]
                z_mu[0, j] = y_values[yv]
                z_mu = torch.Tensor(z_mu)
                x_mean = model.sample(z = z_mu.to(device))
                x_pad = np.pad(x_mean[0].detach().cpu().numpy().reshape(84,84), ((1, 1), (1,1)), 'constant',  constant_values=(1, 1))
                canvas[yv*(h+2):(yv+1)*(h+2), xv*(w+2):(xv+1)*(w+2)] = x_pad
        axs[p].imshow(canvas, origin="upper", cmap="gray")
        axs[p].axis('off')
        axs[p].set_title('reconst between {} and {}'.format(i, j))

        p+=1

    plt.tight_layout()
    save_name = os.path.join(save_path, 'Pairwise_linea_changes_{}.png'.format(epoch))
    fig.savefig(save_name)
    plt.close()



    nx  = 10
    ny = 3

    canvas = np.empty(((h+2)*ny, (w+2)*nx))

    for j in range(ny):
        x_values = np.linspace(z_min[j], z_max[j], nx)
        for i in range(nx):
            z_prepre = z_mean.copy()
            z_prepre[0, j] = x_values[i]
            x_mean = model.sample(z=torch.Tensor(z_prepre).to(device))
            x_pad = np.pad(x_mean.detach().cpu().numpy().reshape(h,w), ((1,1), (1,1)), 'constant', constant_values = (1,1))
            canvas[j*(h+2):(j+1)*(h+2), i*(w+2):(i+1)*(w+2)] = x_pad
    fig = plt.figure(figsize=(10, 4))
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()
    plt.axis('off')
    save_name = os.path.join(save_path, 'Components_{}.png'.format(epoch))
    fig.savefig(save_name)
    plt.close()
