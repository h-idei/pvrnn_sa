"""
Non top (highest) layer of a multi-lalyerd PV-RNN which
- has vertical connection between layers
- is compatible with mini-batch learning
- samples prior from unit gaussian prior at t=0
"""

import torch
import torch.nn as nn
import numpy as np

class PVRNNLayer(nn.Module):
    def __init__(self, d_size: int, z_size: int, tau: int, w: float, w1: float, wd1: float, input_size: int, minibatch_size: int, n_minibatch: int, seq_len: int, layer_name="association", device="cpu"):
        super(PVRNNLayer, self).__init__()
        # hyper parameters - mostly the same as PVRNNTopLayer
        self.d_size         = d_size
        self.z_size         = z_size
        self.tau = torch.zeros(d_size, device=device)
        self.tau[:] = tau
        self.tau[:int(0.5*d_size)] = 2*tau
        print(self.tau.detach().numpy())
        self.w              = w
        self.w1             = w1
        self.wd1             = wd1
        self.input_size     = input_size # dim of a higher layer
        self.minibatch_size = minibatch_size
        self.n_minibatch    = n_minibatch
        self.seq_len        = seq_len
        self.device         = device

        # weights
        self.d_to_h  = nn.Linear(d_size, d_size, bias=False)
        self.z_to_h  = nn.Linear(z_size, d_size, bias=False)
        self.d_to_z  = nn.Linear(d_size, 2 * z_size, bias=False)
        self.hd_to_h = nn.Linear(input_size, d_size, bias=False) # mapping input from higher layer
        
        K = 10
        self.bias_d = torch.empty(d_size, device=device).normal_(mean=0,std=np.sqrt(K))
        while np.abs(np.var(self.bias_d.detach().numpy()) - K) > 0.1 * K :
                self.bias_d = torch.empty(d_size, device=device).normal_(mean=0,std=np.sqrt(K))
        # "a"
        self.A  = nn.ParameterList(nn.Parameter(torch.zeros(minibatch_size, seq_len, 2 * z_size, device=device)) for i in range(n_minibatch))
        self.A0 = torch.zeros(minibatch_size, 2 * z_size, device=device)
        
        #set shape of prior
        self.in_p = torch.zeros(minibatch_size, seq_len, 2 * z_size, device=device)

        
        # initial state of deterministic variable
        self.d0 = torch.zeros(minibatch_size, d_size, device=device)
        self.h0 = torch.zeros(minibatch_size, d_size, device=device)
        
        # if you optimize initial states of deterministic variables
        self.init_h = nn.ParameterList(nn.Parameter(torch.zeros(minibatch_size, d_size, device=device)) for i in range(n_minibatch))
        self.init_h_mu = nn.ParameterList(nn.Parameter(torch.zeros(d_size, device=device)) for i in range(n_minibatch))

    #this part may be used as one way of calculating like complexity term if you want to optimize initial states of deterministic variables ref"Learning to Perceive the World as Probabilistic or Deterministic via Interaction With Others: A Neuro-Robotics Experiment"[Murata et al., 2017]
    def compute_nll_init_h(self):
        init_var = 1.0 #set varince
        self.nll_init_h = torch.sum(0.5 * torch.log(2.0 * torch.pi * init_var) + 0.5 * (self.init_h[self.minibatch_ind] - self.init_h_mu[self.minibatch_ind]) * (self.init_h[self.minibatch_ind] - self.init_h_mu[self.minibatch_ind]) / init_var) / self.d_size
    
    def initialize(self, minibatch_ind: int):
        self.t = 0
        self.minibatch_ind = minibatch_ind
        if self.wd1 != 0.0: 
            self.h = self.init_h[self.minibatch_ind][:,:]
            self.d = torch.tanh(self.h)
        else:
            self.h = self.h0
            self.d = self.d0
        self.compute_nll_init_h()
        self.wnll_init_h = self.wd1 * self.nll_init_h
        
    def compute_mu_sigma(self, epo: int):
        if self.t == 0:
            # prior is unit gaussian at t=1
            mu_in_p, lnsigma_p = self.in_p[:, self.t, :].chunk(2, 1) #in_p = 0
            self.mu_p = torch.tanh(mu_in_p) #tanh(0)=0
            self.sigma_p = torch.exp(lnsigma_p) #exp(0)=1
            
            #initial posterior is initialized by prior
            if epo == 0:
                self.A[self.minibatch_ind].data[:, self.t, :] = torch.cat([mu_in_p.clone().detach(), lnsigma_p.clone().detach()], dim=1)
            
            mu_in_q, lnsigma_q = self.A[self.minibatch_ind][:, self.t, :].chunk(2, 1)
            self.mu_q = torch.tanh(mu_in_q)
            self.sigma_q = torch.exp(lnsigma_q)


        elif 0 < self.t < self.seq_len:
        
            mu_in_p, lnsigma_p = self.d_to_z(self.d).chunk(2, 1)
            self.mu_p    = torch.tanh(mu_in_p)
            self.sigma_p = torch.exp(lnsigma_p)
            
            #initial posterior is initialized by prior
            if epo == 0:
                self.A[self.minibatch_ind].data[:, self.t, :] = torch.cat([mu_in_p.clone().detach(), lnsigma_p.clone().detach()], dim=1)
              
            mu_in_q, lnsigma_q = self.A[self.minibatch_ind][:, self.t, :].chunk(2, 1)
            self.mu_q    = torch.tanh(mu_in_q)
            self.sigma_q = torch.exp(lnsigma_q)

        else:
            mu_in_p, lnsigma_p = self.d_to_z(self.d).chunk(2, 1)
            self.mu_p    = torch.tanh(mu_in_p)
            self.sigma_p = torch.exp(lnsigma_p)

    def sample_zq(self):
        self.z = self.mu_q + torch.randn(self.sigma_q.shape) * self.sigma_q

    def sample_zp(self):
        self.z = self.mu_p + torch.randn(self.sigma_p.shape) * self.sigma_p

    def compute_mtrnn(self, hd):
        # consider the input from a higher layer as well
        h = (1 - 1 / self.tau) * self.h + (self.d_to_h(self.d) + self.z_to_h(self.z) + self.hd_to_h(hd) + self.bias_d) / self.tau
        d = torch.tanh(h)
        self.h = h
        self.d = d

    def compute_kl(self):
        eps = 0.000001 #to avoid null computation
        self.kl = torch.sum(torch.log(self.sigma_p + eps) - torch.log(self.sigma_q + eps)
                       - 0.5 + 0.5 * (self.mu_q.pow(2) + self.sigma_q.pow(2)
                                      - 2 * self.mu_q * self.mu_p + self.mu_p.pow(2)) / (self.sigma_p.pow(2) + eps)) / self.z_size

    
    def posterior_step(self, epo: int, hd):
        self.compute_mu_sigma(epo)
        self.sample_zq()
        self.compute_mtrnn(hd)
        self.compute_kl()
        if self.t == 0:
            self.wkl = self.w1 * self.kl
        else:
            self.wkl = self.w * self.kl
        self.t += 1

    def prior_step(self, hd):
        self.compute_mu_sigma()
        self.sample_zp()
        self.compute_mtrnn(hd)
        self.t += 1

