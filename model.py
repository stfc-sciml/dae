#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 2023
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def stretch(X, alpha, gamma, beta, moving_mag, moving_min, eps, momentum, training):
    '''
    the code is based on the batch normalization in
    http://preview.d2l.ai/d2l-en/master/chapter_convolutional-modern/batch-norm.html
    '''
    if not training:
        X_hat = (X - moving_min)/moving_mag
    else:
        assert len(X.shape) in (2, 4)
        min_ = X.min(dim=0)[0]
        max_ = X.max(dim=0)[0]

        mag_ = max_ - min_
        X_hat =  (X - min_)/mag_
        moving_mag = momentum * moving_mag + (1.0 - momentum) * mag_
        moving_min = momentum * moving_min + (1.0 - momentum) * min_
    Y = (X_hat*gamma*alpha) + beta
    return Y, moving_mag.data, moving_min.data




class Stretch(nn.Module):
    '''
    the code is based on the batch normalization in
    http://preview.d2l.ai/d2l-en/master/chapter_convolutional-modern/batch-norm.html
    '''
    def __init__(self, num_features, num_dims, alpha):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.alpha = alpha
        self.gamma = nn.Parameter(0.01*torch.ones(shape))
        self.beta = nn.Parameter(np.pi*torch.ones(shape))
        self.register_buffer('moving_mag', 1.*torch.ones(shape))
        self.register_buffer('moving_min', np.pi*torch.ones(shape))

    def forward(self, X):
        if self.moving_mag.device != X.device:
            self.moving_mag = self.moving_mag.to(X.device)
            self.moving_min = self.moving_min.to(X.device)
        Y, self.moving_mag, self.moving_min = stretch(
            X, self.alpha , self.gamma, self.beta, self.moving_mag, self.moving_min,
            eps=1e-5, momentum=0.99, training = self.training)
        return Y



class Conv_BN_LRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv_BN_LRelu,self).__init__()
        self.layers =  nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.LeakyReLU())

    def forward(self,x):
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        return  x

class ConvT_BN_LRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvT_BN_LRelu,self).__init__()
        self.layers =  nn.ModuleList()
        self.layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding))
        self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(nn.LeakyReLU())

    def forward(self,x):
        for idx in range(len(self.layers)):
            x = self.layers[idx](x)
        return  x


class DGAECONV(nn.Module):
    def __init__(self, params):
        super(DGAECONV, self).__init__()
        self.input_dim, self.hidden_dims, self.fc_hidden, self.latent_dim, self.alpha  = params

        _, m, n  = self.input_dim


        self.encoder_seq = nn.ModuleList()

        for idx in range(len(self.hidden_dims)-1):
            self.encoder_seq.append(Conv_BN_LRelu(self.hidden_dims[idx], self.hidden_dims[idx+1], 10, 4,
                                              padding = 1))


        self.en_fc = nn.Linear(self.hidden_dims[-1]*4*4, self.fc_hidden)
        self.to_lat = nn.Linear(self.fc_hidden, self.latent_dim)
        self.strecth = Stretch(self.latent_dim, 2, self.alpha)

        self.to_dec = nn.Linear(self.latent_dim*2, self.fc_hidden)
        self.de_fc = nn.Linear(self.fc_hidden, self.hidden_dims[-1]*4*4)


        self.rhidden_dims = self.hidden_dims[::-1]

        self.decoder_seq = nn.ModuleList()

        for idx in range(len(self.rhidden_dims)-1):
            self.decoder_seq.append(ConvT_BN_LRelu(self.rhidden_dims[idx], self.rhidden_dims[idx+1], 10, 4,
                                                       padding = 1))

        self.decoder_seq.append(nn.Conv2d(self.rhidden_dims[-1], self.rhidden_dims[-1], 3,
                                                   padding = 1))
        self.decoder_seq.append(nn.Sigmoid())



    def sample(self, num_samples = 100, z = None):
        c = torch.cat((torch.cos(2*np.pi*z), torch.sin(2*np.pi*z)), 0)
        c = c.T.reshape(self.latent_dim*2, -1).T
        samples = self.decode(c)
        return samples

    def reconstr(self, x):
        z = self.encode(x)
        c = torch.cat((torch.cos(2*np.pi*z), torch.sin(2*np.pi*z)), 0)
        c = c.T.reshape(self.latent_dim*2, -1).T
        reconstr = self.decode(c)
        return reconstr

    def encode(self, x):
        for idx in range(len(self.encoder_seq)):
            x = self.encoder_seq[idx](x)

        x = torch.flatten(x, start_dim=1)

        x = self.en_fc(x)

        z = self.to_lat(x)
        s = self.strecth(z)

        return s

    def latent(self, x):
        z = self.encode(x)
        return z

    def decode(self, x):
        x = nn.LeakyReLU()(self.to_dec(x))
        x = nn.LeakyReLU()(self.de_fc(x))
        x = x.view(-1, self.hidden_dims[-1], 4, 4)

        for idx in range(len(self.decoder_seq)):
            x = self.decoder_seq[idx](x)
        return x

    def reparameterize(self, z):
        diff = torch.abs(z - z.unsqueeze(axis = 1))
        none_zeros = torch.where(diff == 0., torch.tensor([100.]).to(z.device), diff)
        z_scores,_ = torch.min(none_zeros, axis = 1)
        std =  torch.normal(mean = 0., std = 1.*z_scores).to(z.device)
        s = z + std
        c = torch.cat((torch.cos(2*np.pi*s), torch.sin(2*np.pi*s)), 0)
        c = c.T.reshape(self.latent_dim*2,-1).T
        return c

    def forward(self, x):
        z = self.encode(x)
        c = self.reparameterize(z)
        reconstr = self.decode(c)
        return [reconstr, c, z]
