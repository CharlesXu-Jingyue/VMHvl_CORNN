#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:48:03 2023

@author: dinc
"""

import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time as time_now
from rnn_class import RNN
from utils_admm import solve_corrn_admm_gpu
from utils_admm import solve_corrn_admm
from sklearn.metrics import r2_score

# Define the network
class Model(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, alpha=0.1):
        super(Model, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.alpha = alpha

        # Initialize weights
        self.W_in = nn.Parameter(torch.randn(hidden_dims, input_dims).to('cuda'))
        self.W_rec = nn.Parameter(torch.randn(hidden_dims, hidden_dims).to('cuda'))
        self.W_out = nn.Parameter(torch.randn(output_dims, hidden_dims).to('cuda'))
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_rec)
        nn.init.xavier_uniform_(self.W_out)

        # Set W_rec diagonal to zero
        self.W_rec.data = self.W_rec.data * (1 - torch.eye(hidden_dims).to('cuda'))

    def forward(self, u, r, noise_in=0.1, noise_con=0):
        """
        Inputs
        The shape of u is (seq_len, input_dims)
        The shape of r is (hidden_dims), initialization of hidden dims
        
        Outputs
        The shape of o is (seq_len, output)
        The shape of hidden_states is (seq_len+1, hidden_dims)
        """
        T = u.shape[0]  # Sequence length
        n_rec = r.shape[0]  # Number of hidden dimensions
        
        hidden_states = torch.zeros([T+1, n_rec]).to('cuda')
        hidden_states[0, :] = r.flatten().to('cuda')
        x = torch.arctanh(r).to('cuda')
        o = torch.zeros([T, self.output_dims]).to('cuda')
        for t in range(T):
            x = (1 - self.alpha) * x + self.alpha * (self.W_rec @ r + self.W_in @ u[t, :] \
                + torch.normal(torch.zeros(n_rec), noise_in).to('cuda'))
            r = torch.tanh(x).to('cuda') + torch.normal(torch.zeros(n_rec), noise_con).to('cuda')
            hidden_states[t+1, :] = r
            o[t, :] = self.W_out @ r.flatten()
                
        return hidden_states, o

num_trials = 30             # number of trials
n_in = 1                    # input dimensions
n_rec = 500                 # number of hidden dimensions
T = 1000                    # sequence length
width = 20                  # Gaussian width
gaus_int = 500              # Gaussian intensity

# Initialize arrays to store results
r_in_all = np.zeros([T*num_trials, n_rec])
r_out_all = np.zeros([T*num_trials, n_rec])
u_all = np.zeros([T*num_trials, n_in])

# Load the model
model = Model(n_in, n_rec, n_in)
# model = torch.load('models/model_0.pt', weights_only=False)

for i in range(num_trials):
    u = torch.zeros([T, n_in]).to('cuda')
    u[0:100, 0] = 1
    out_gt = torch.tensor(np.exp(-((np.arange(T) - gaus_int)**2)/(2*width**2))).to('cuda')
    
    r_init = torch.rand(n_rec).to('cuda') - 0.5

    # Forward pass
    h, out = model(u, r_init)
    
    r = h.detach().cpu().numpy()
    out = out.detach().cpu().numpy()
    u = u.detach().cpu().numpy()
    gnd = model.W_rec.data.detach().cpu().numpy().flatten()
    u_all[i*T:(i+1)*T, :] = u
    r_in_all[i*T:(i+1)*T, :] = r[:-1, :]
    r_out_all[i*T:(i+1)*T, :] = r[1:, :]
        
    print(i)

n_sup = 500 # number of supervised dimensions
nt = 30 # number of trials for supervised learning
r_in = r_in_all[:T*nt, :n_sup]
r_out = r_out_all[:T*nt, :n_sup]
u_in = u_all[:T*nt, :]

# Ground truth weights
gnd = model.W_rec.data.detach().cpu().numpy()
gnd = gnd[:n_sup, :n_sup].flatten()
alph = 0.1 + np.random.normal(0, 0, n_sup)
w = solve_corrn_admm(r_in, r_out, u_in=u_in, alph=alph, l2=1e-5, 
                     threshold=1, rho=1, verbose=2, num_iters=0,
                     gnd=gnd, solver_type='weighted')

w_rec_cornn = w[:n_sup, :].T
w_in_cornn = w[n_sup:, :].T

w_rec_cornn[np.eye(n_sup, dtype=bool)] = 0

# Plot and save the scatter plot for recurrent weights
prd = w_rec_cornn.flatten()
plt.scatter(gnd, prd)
plt.title('Recurrent Weights Correlation')
plt.xlabel('Ground Truth')
plt.ylabel('Predicted')
plt.savefig('figures/recurrent_weights_correlation.png')
plt.close()
print(pearsonr(gnd, prd))

# Plot and save the scatter plot for input weights
gnd_in = model.W_in.data.detach().cpu().numpy()[:n_sup]
prd_in = w_in_cornn.flatten()
plt.scatter(gnd_in, prd_in)
plt.title('Input Weights Correlation')
plt.xlabel('Ground Truth')
plt.ylabel('Predicted')
plt.savefig('figures/input_weights_correlation.png')
plt.close()
print(pearsonr(gnd_in.flatten(), prd_in))

opts = {}
opts['n_rec'] = n_sup
opts['n_in'] = 3
opts['alpha'] = alph
opts['verbose'] = False
opts['sigma_input'] = 0
opts['sigma_conversion'] = 0 
m1 = RNN(opts)
m1.rnn['w_rec'] = w_rec_cornn
m1.rnn['w_in'] = w_in_cornn

u = torch.zeros([T, n_in]).to('cuda')
u[0:100, 0] = 1
u[400:410, 0] = 10
out_gt = torch.tensor(np.exp(-((np.arange(T) - gaus_int)**2)/(2*width**2))).to('cuda')

# Forward pass
r_init = torch.rand(n_rec).to('cuda') - 0.5
h, out = model(u, r_init, 0)

r_gt = h.detach().cpu().numpy()
out_gt = out.detach().cpu().numpy()
u = u.detach().cpu().numpy()
r_cornn = m1.get_time_evolution(T=u.shape[0], u=u, r_in=r_gt[0, :n_sup])

pick = 0
plt.plot(r_cornn[:, pick], label='CORNN')
plt.plot(r_gt[:, pick], label='Ground Truth')
plt.title('Time Evolution of Neuron {}'.format(pick))
plt.xlabel('Time')
plt.ylabel('Activation')
plt.legend()
plt.savefig('figures/time_evolution_neuron_{}.png'.format(pick))
plt.close()

print(r2_score(r_gt[:, :n_sup].flatten(), r_cornn.flatten()))
