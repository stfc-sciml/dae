from __future__ import print_function
import os
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from model import DGAECONV
from dataset import XYC
from plot import draw_recover

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Disentangling Autoencoder')
parser.add_argument('--data_path', type=str, default='datasets', help='data folder path')
parser.add_argument('--save_path', type=str, default='reuslts', help='save folder path')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10000, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--seed', type=int, default=100, help='random seed to use. Default=123')
opt = parser.parse_args()

print(opt)

try:
    os.mkdir(opt.save_path)
except OSError:
    pass

np.random.seed(opt.seed)
torch.manual_seed(opt.seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


print('===> Loading datasets')
train_x = XYC(opt.data_path)    
test_x = XYC(opt.data_path)    
dataset_size = train_x.__len__()

train_dataloader = DataLoader(train_x, batch_size= opt.batchSize, shuffle=True)
test_dataloader = DataLoader(test_x, batch_size= opt.testBatchSize, shuffle=True)

repre = train_x[7029]['x']
repre = repre.unsqueeze(0)


print('===> Building model')
W = [[1., 1., 0.001]]
model = DGAECONV(((1, 84, 84), (1, 8, 16), 64, 3, torch.Tensor(W).to(device))).to(device)

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.5))


def train(epoch):
    model.train() 
    epoch_loss = 0
    for iteration, batch in enumerate(train_dataloader, 1):
        input, target = batch['x'].to(device), batch['x'].to(device)

        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output[0], target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(train_dataloader)))


def checkpoint(epoch):
    model_out_path = os.path.join(opt.save_path, "model_epoch_{}.pth".format(epoch))
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(opt.nEpochs):
    train(epoch)
    
    if epoch % 50 == 0:
        checkpoint(epoch)
        draw_recover(opt.save_path, model, test_dataloader, device, epoch)
    
    
checkpoint(opt.nEpochs)
draw_recover(opt.save_path, model, test_dataloader, device, opt.nEpochs)


