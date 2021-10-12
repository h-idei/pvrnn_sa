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
    def __init__(self, d_size: int, z_size: int, tau: int, w: float, w1: float, wd1: float, input_size: int, bias, minibatch_size: int, n_minibatch: int, seq_len: int, layer_name="association", device="cpu"):
        super(PVRNNLayer, self).__init__()
        # hyper parameters - mostly the same as PVRNNTopLayer
        self.d_size         = d_size
        self.z_size         = z_size
        self.tau = torch.zeros(d_size, device=device)
        self.tau[:] = tau
        self.tau[:int(0.5*d_size)] = 2*tau # multiple timescales
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
        
        #set trained bias
        self.bias_d = bias

        # "a"
        self.A  = nn.ParameterList(nn.Parameter(torch.zeros(minibatch_size, seq_len, 2 * z_size, device=device)) for i in range(n_minibatch))
        self.A0 = torch.zeros(minibatch_size, 2 * z_size, device=device)
        
        
        # initial state of deterministic variable
        self.d0 = torch.zeros(minibatch_size, d_size, device=device)
        self.h0 = torch.zeros(minibatch_size, d_size, device=device)
        
        #if you optimize initial states of deterministic variables
        self.init_h = nn.ParameterList(nn.Parameter(torch.zeros(minibatch_size, d_size, device=device)) for i in range(n_minibatch))
        self.init_h_mu = nn.ParameterList(nn.Parameter(torch.zeros(d_size, device=device)) for i in range(n_minibatch))
        
    #this part may be used as one way of calculating like complexity term if you optimize initial states of deterministic variables ref"Learning to Perceive the World as Probabilistic or Deterministic via Interaction With Others: A Neuro-Robotics Experiment"[Murata et al., 2017]
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

    
    def posterior_step(self, hd):
        self.compute_mu_sigma()
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

    def er_initialize(self, window, situation):
        self.window = window
        #Initial A value is initialized by learned parameter according to switch pattern
        if situation == 0:
            A_midian, idx = torch.median(self.A[0].clone().detach()[24:, 0, :], 0)
        else:
            A_midian, idx = torch.median(self.A[0].clone().detach()[:24, 0, :], 0)
        self.er_A = nn.Parameter(torch.cat((torch.reshape(A_midian, (1, 2 * self.z_size)), torch.zeros(window-1, 2 * self.z_size))).to(self.device))
        self.er_A0  = torch.zeros(2 * self.z_size, device=self.device)
        self.d_last_state = torch.zeros(self.d_size, device=self.device)
        self.h_last_state = torch.zeros(self.d_size, device=self.device)
        self.initial_window = True
        self.t = 0

    def er_slide_window(self):
        self.er_A = nn.Parameter(torch.cat((self.er_A[1:].clone().detach(), torch.randn(1, 2 * self.z_size))).to(self.device))
        self.initial_window = False

    def er_compute_mu_sigma(self, itr):    
        if self.initial_window and self.t == 0:
            mu_in_p, lnsigma_p = self.d_to_z(self.d).chunk(2) # initial d = 0
            mu_in_q, lnsigma_q = self.er_A[self.t, :].chunk(2) 
            self.mu_p = torch.zeros_like(mu_in_p)
            self.sigma_p = torch.ones_like(lnsigma_p)
            self.mu_q = torch.tanh(mu_in_q)
            self.sigma_q = torch.exp(lnsigma_q)
        elif self.t < self.window:

            mu_in_p, lnsigma_p = self.d_to_z(self.d).chunk(2)
            
            if self.initial_window and itr == 0:
                self.er_A.data[self.t, :] = torch.cat([mu_in_p.clone().detach(), lnsigma_p.clone().detach()], dim=0)
            elif itr == 0 and self.t == self.window - 1:
                self.er_A.data[self.window - 1, :] = torch.cat([mu_in_p.clone().detach(), lnsigma_p.clone().detach()], dim=0)

            mu_in_q, lnsigma_q = self.er_A[self.t, :].chunk(2)
            self.mu_p    = torch.tanh(mu_in_p)
            self.sigma_p = torch.exp(lnsigma_p)
            self.mu_q    = torch.tanh(mu_in_q)
            self.sigma_q = torch.exp(lnsigma_q)
        else:

            mu_in_p, lnsigma_p = self.d_to_z(self.d).chunk(2)
            self.mu_p    = torch.tanh(mu_in_p)
            self.sigma_p = torch.exp(lnsigma_p)

    def er_step(self, hd, itr):
        self.er_compute_mu_sigma(itr)
        self.sample_zq()
        self.compute_mtrnn(hd)
        self.compute_kl()
        if self.initial_window & self.t == 0:
            self.wkl = self.w1 * self.kl
        else:
            self.wkl = self.w * self.kl
        self.t += 1

    def er_prior_step(self, hd):
        self.er_compute_mu_sigma(1)
        self.sample_zp()
        self.compute_mtrnn(hd)
        self.t += 1

    def er_keep_last_state(self):
        self.d_last_state = self.d
        self.h_last_state = self.h

    def er_reset(self):
        self.d = self.d_last_state
        self.h = self.h_last_state
        self.t = 0

