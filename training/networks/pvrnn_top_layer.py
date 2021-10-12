"""
Executive (highest) layer of a multi-lalyerd PV-RNN which
- has vertical connection between layers
- is compatible with mini-batch learning
- samples prior from unit gaussian prior at t=0
"""

import torch
import torch.nn as nn
import numpy as np


class PVRNNTopLayer(nn.Module):
    def __init__(self, d_size: int, z_size: int, tau: int, w: float, w1: float, wd1: float, minibatch_size: int, n_minibatch: int, seq_len: int, device="cpu"):
        super(PVRNNTopLayer, self).__init__()
        """hyper parameters:
        d_size:         # d neurons in this layer
        z_size:         # z neurons in this layer
        tau:            time-constant used in MTRNN computation in this layer
        w:              the value of meta-prior from t=2
        w1:             the value of meta-prior at t=1
        minibatch_size: # sequences in minibatch
        n_minibatch:    # minibatches
        seq_len:        # time steps in a sequence
        device:         whether the network is on cpu or gpu
        
        Note that minibatch_size, n_minibatch, and seq_len are only used during training
        """
        self.d_size     = d_size
        self.z_size     = z_size
        self.tau        = tau
        self.w          = w
        self.w1         = w1
        self.wd1         = wd1
        self.batch_size = minibatch_size
        self.n_minibatch = n_minibatch
        self.seq_len    = seq_len
        self.device     = device
        
        K = 10 # variance of bias
        self.bias_d = torch.empty(d_size, device=device).normal_(mean=0,std=np.sqrt(K))
        while np.abs(np.var(self.bias_d.detach().numpy()) - K) > 0.1 * K :
                self.bias_d = torch.empty(d_size, device=device).normal_(mean=0,std=np.sqrt(K))

        """initialize "a" term for approximate posterior computation
        The network holds "a" for all the training sequences as a nn.ParameterList where each item is a nn.Parameter representaing a vector concatenating a^{\mu} and a^{\sigma} for all time steps.
        This vector has the dimension of n_minibatch x seq_len x 2*z_size where 2 means a pair of mu and sigma and each element is picked up when computing approximate posterior
        # items in the list corresponds to # minibatch 
        """
        
        #initial prior and posterior is unit gaussian (mu:tanh(0)=0, sigma:exp(0)=1)
        self.A = nn.ParameterList(nn.Parameter(torch.zeros(minibatch_size, seq_len, 2 * z_size, device=device)) for i in range(n_minibatch))
        self.in_p = torch.zeros(minibatch_size, seq_len, 2 * z_size, device=device)
        
        # initial deterministic states (hidden state at t=0) fixed to zero vectors in PV-RNN
        self.d0 = torch.zeros(minibatch_size, d_size, device=device)
        self.h0 = torch.zeros(minibatch_size, d_size, device=device)
        self.init_h = nn.ParameterList(nn.Parameter(torch.zeros(minibatch_size, d_size, device=device)) for i in range(n_minibatch))
        self.init_h_mu = nn.ParameterList(nn.Parameter(torch.zeros(d_size, device=device)) for i in range(n_minibatch))
    
    
    def compute_nll_init_h(self):
        init_var = 1.0 #set varince
        self.nll_init_h = torch.sum(0.5 * torch.log(2.0 * torch.pi * init_var) + 0.5 * (self.init_h[self.minibatch_ind] - self.init_h_mu[self.minibatch_ind]) * (self.init_h[self.minibatch_ind] - self.init_h_mu[self.minibatch_ind]) / init_var) / self.d_size
    
    # initialize the network state before forward computation and receive the index of minibatch
    def initialize(self, minibatch_ind: int):
        self.t = 0
        self.minibatch_ind = minibatch_ind
        #set initial deterministic state
        if self.wd1 != 0.0:
            self.h = self.init_h[self.minibatch_ind][:,:]
            self.d = torch.tanh(self.h)

        else:
            self.h = self.h0 #zero
            self.d = self.d0 #zero
        self.compute_nll_init_h()
        self.wnll_init_h = self.wd1 * self.nll_init_h
        

    # compute mu and sigma for both prior and posterior in this function
    def compute_mu_sigma(self):
        if self.t == 0:
            # map mu_q, sigma_q, mu_p, sigma_p with one tensor and split them
            self.mu_in_p, self.lnsigma_p = self.in_p[:, self.t, :].chunk(2, 1) #in_p=0
            self.mu_in_q, self.lnsigma_q = self.A[self.minibatch_ind][:, self.t, :].chunk(2, 1)

            # prior is unit gaussian at t=1
            self.mu_p = torch.tanh(self.mu_in_p)
            self.sigma_p = torch.exp(self.lnsigma_p)
            
            self.mu_q = torch.tanh(self.mu_in_q)
            self.sigma_q = torch.exp(self.lnsigma_q)
            

        elif 0 < self.t < self.seq_len:
     
            mu_in_q = self.mu_in_q
            lnsigma_q = self.lnsigma_q

            self.mu_q = torch.tanh(mu_in_q)
            self.sigma_q = torch.exp(lnsigma_q)

        else:

            self.mu_p = torch.zeros_like(self.mu_in_q)
            self.sigma_p = torch.ones_like(self.lnsigma_q)
        

    def sample_zq(self):
        self.z = self.mu_q + torch.randn(self.sigma_q.shape) * self.sigma_q # reparameterization

    def sample_zp(self):
        self.z = self.mu_p + torch.randn(self.sigma_p.shape) * self.sigma_p

    def compute_mtrnn(self):
        # since this is the executive (highest) layer, it doesn't use h calculated below
        h = (1 - 1 / self.tau) * self.h + (self.d_to_h(self.d) + self.z_to_h(self.z) + self.bias_d) / self.tau
        d = torch.tanh(h)
        self.h = h
        self.d = d

    def compute_kl(self):
        eps = 0.000001 #to avoid null computation
        self.kl = torch.sum(torch.log(self.sigma_p + eps) - torch.log(self.sigma_q + eps)
                       - 0.5 + 0.5 * (self.mu_q.pow(2) + self.sigma_q.pow(2)
                                      - 2 * self.mu_q * self.mu_p + self.mu_p.pow(2)) / (self.sigma_p.pow(2) + eps)) / self.z_size 
    
    def posterior_step(self):
        self.compute_mu_sigma()
        self.sample_zq()
        self.compute_kl()
        # use w and w1 differently depending on time step
        if self.t == 0:
            self.wkl = self.w1 * self.kl
        else:
            self.wkl = self.w * self.kl
        self.t  += 1

    def prior_step(self):
        self.compute_mu_sigma()
        self.sample_zq() #self.sample_zp()
        self.t += 1

