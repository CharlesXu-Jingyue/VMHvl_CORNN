#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:32:18 2022
@author: dinc

Modified: 20250213
By Charles Xu @ Caltech
"""

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time as time_now
import os

# Define the network
class Model(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, alpha=0.1):
        super(Model, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.alpha = torch.tensor(alpha).to('cuda')

        # initialize weights
        self.W_in = nn.Parameter(torch.randn(hidden_dims, input_dims))
        self.W_rec = nn.Parameter(torch.randn(hidden_dims, hidden_dims))
        self.W_out = nn.Parameter(torch.randn(output_dims, hidden_dims))
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_rec)
        nn.init.xavier_uniform_(self.W_out)

        # set W_rec diagonal to zero
        self.W_rec.data = self.W_rec.data * (1 - torch.eye(hidden_dims))
        self.to('cuda')

    def forward(self, u, r, noise=0.1):
        """
        Inputs
        The shape of u is (seq_len, input_dims)
        The shape of r is (hidden_dims), initialization of hidden dims
        
        Outputs
        The shape of o is (seq_len, output)
        The shape of hidden_states is (seq_len+1, hidden_dims)
        """
        T = u.shape[0]  # sequence length
        n_rec = r.shape[0]  # number of hidden dimensions
        
        hidden_states = torch.zeros([T+1, n_rec]).to('cuda')
        hidden_states[0, :] = r.flatten().to('cuda')
        x = torch.arctanh(r).to('cuda')
        o = torch.zeros([T, self.output_dims]).to('cuda')
        for t in range(T):
            x = (1 - self.alpha) * x + self.alpha * (self.W_rec @ r + self.W_in @ u[t, :] \
                + torch.normal(torch.zeros(n_rec), noise).to('cuda'))
            r = torch.tanh(x)
            hidden_states[t+1, :] = r
            o[t, :] = self.W_out @ r.flatten()
                
        return hidden_states, o

# Define hyperparameters
n_in = 1                # input dimensions
n_rec = 500             # hidden dimensions
n_reg = 200             # not used in the code
num_trials = 1500       # number of training trials
T = 600                 # sequence length
width = 20              # width of the Gaussian
gaus_int = T - 100      # center of the Gaussian
num_exps = 1            # number of experiments

if not os.path.exists('figures'):
    os.makedirs('figures')
if not os.path.exists('models'):
    os.makedirs('models')

for k in range(num_exps):
    # instantiate the model
    model = Model(n_in, n_rec, n_in)
    
    # define the mean squared error loss function
    criterion = nn.MSELoss()
    
    # define the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-7)
    
    # train the model
    for epoch in range(num_trials):
        u = torch.zeros([T, n_in]).to('cuda')  # input sequence
        out_gt = torch.zeros([T, n_in]).to('cuda')  # ground truth output
        r_in = torch.rand(n_rec).to('cuda') - 0.5  # initial hidden state
    
        u[0:100, 0] = 1  # set the first 100 time steps of the input to 1
        out_gt[:, 0] = torch.tensor(np.exp(-((np.arange(T) - gaus_int)**2) / (2 * width**2))).to('cuda')
        
        # forward pass
        h, out = model(u, r_in) # hidden states and output
    
        # compute the loss
        loss = criterion(out, out_gt)
        
        # backward pass
        loss.backward()
        
        # update the weights
        optimizer.step()
        
        # Ensure the diagonal is zero 
        model.W_rec.data = model.W_rec.data * (1 - torch.eye(n_rec).to('cuda'))
        
        # zero the gradients
        optimizer.zero_grad()
        
        # print the loss and plot the results every 300 epochs
        if epoch % 300 == 0:
            plt.figure(figsize=(10, 15))
            temp = time_now.localtime()
            current_time = time_now.strftime("%H:%M:%S", temp)
            print('{}: epoch {}: loss {}'.format(current_time, epoch, criterion(out, out_gt).item()))

            r_in = torch.rand(n_rec).to('cuda') - 0.5
            u = torch.zeros([T, n_in]).to('cuda')

            u[0:100, 0] = 1
            h_no, out = model(u, r_in) # hidden states and output without additional input
                        
            plt.subplot(5, 1, 4)
            plt.plot(out.cpu().detach().numpy()[:, 0])
            plt.plot(out_gt.cpu().detach().numpy()[:, 0])
            plt.title('Output without additional input')

            u[300:350, 0] = 0.5
            h_yes, out = model(u, r_in) # hidden states and output with additional input
            
            plt.subplot(5, 1, 5)
            plt.plot(out.cpu().detach().numpy()[:, 0])
            plt.plot(out_gt.cpu().detach().numpy()[:, 0])
            plt.title('Output with additional input')
            
            h_no = h_no.cpu().detach().numpy() # hidden states without additional input
            h_yes = h_yes.cpu().detach().numpy() # hidden states with additional input
            
            plt.subplot(5, 1, 1)
            plt.plot(h_no[:, 0])
            plt.plot(h_yes[:, 0])
            plt.title('Hidden state 0')

            plt.subplot(5, 1, 2)
            plt.plot(h_no[:, 2])
            plt.plot(h_yes[:, 2])
            plt.title('Hidden state 2')
            
            plt.subplot(5, 1, 3)
            plt.plot(h_no[:, 4])
            plt.plot(h_yes[:, 4])
            plt.title('Hidden state 4')
            
            plt.tight_layout()
            plt.savefig(f'figures/epoch_{epoch}.png')
            plt.close()

            print(h_no.shape)
        print('\rEpoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_trials, loss.item()), end='')
    
    torch.save(model, 'models/model_%d.pt' % k)
