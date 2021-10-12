"""
a multi-lalyerd PV-RNN which
- has vertical connection between layers
- is compatible with mini-batch learning
- samples prior from unit gaussian prior at t=0
*In Idei's model, top PV-RNN consists of executive latent and associative module
"""
import sys
sys.path.append("../")

import torch
import torch.nn as nn
from networks.pvrnn_top_layer import PVRNNTopLayer
from networks.pvrnn_layer import PVRNNLayer


class PVRNN(nn.Module):
    def __init__(self, d_size: list, z_size: list, tau: list, w: list, w1: list, wd1: list, bias: list, minibatch_size: int, n_minibatch: int, seq_len, input=False, input_dim=None, device="cpu"):
        super(PVRNN, self).__init__()
        # The following hyper parameters are given as lists in which items follow the order from higher layer to lower layer
        self.d_size         = d_size
        self.z_size         = z_size
        self.tau            = tau
        self.w              = w
        self.w1             = w1
        self.wd1             = wd1  #weight of negative log-likelihood (complexity) term for constraint on initial deterministic state
        self.bias = bias

        # The following is the same as the hyper parameters in layer
        self.minibatch_size = minibatch_size
        self.n_minibatch    = n_minibatch
        self.seq_len    = seq_len
        self.device     = device

        self.input      = input       # if this layer receives any inputs or not
        self.input_dim  = input_dim   # if so, the dimension of the input is given here
        self.n_layer    = len(d_size) # # layers in this network

        self.layers = nn.ModuleList() # this network holds the layers with nn.ModuleList()
        

        # initialize the network parameters
        if input:
            # set sensory modules
            self.layers.append(PVRNNLayer(d_size[0], z_size[0], tau[0], w[0], w1[0], wd1[0], input_dim, bias[0], minibatch_size, n_minibatch, seq_len, "sensory", device))

        else:
            # set top-layer pb-like latent
            self.layers.append(PVRNNTopLayer(d_size[0], z_size[0], tau[0], w[0], w1[0], wd1[0], bias[0], minibatch_size, n_minibatch, seq_len, device))

        # set associative module
        for l in range(1, self.n_layer):
            self.layers.append(PVRNNLayer(d_size[l], z_size[l], tau[l], w[l], w1[l], wd1[l], z_size[l - 1], bias[l], minibatch_size, n_minibatch, seq_len, "association", device))

    # reset the network for forward computation
    def initialize(self, minibatch_ind):
        for layer in self.layers:
            layer.initialize(minibatch_ind)


    def posterior_step(self, x=None):
        # hold the values of KLD and NLL (of initial d state) for each layer as a tensor
        kl  = torch.zeros(self.n_layer, device=self.device)
        wkl = torch.zeros(self.n_layer, device=self.device)
        nll_init_h  = torch.zeros(self.n_layer, device=self.device)
        wnll_init_h = torch.zeros(self.n_layer, device=self.device)
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.posterior_step(x) if self.input else layer.posterior_step()
            else:
                layer.posterior_step(self.layers[i - 1].z) #from executive latent to associative module
            kl[i] = layer.kl
            wkl[i] = layer.wkl
            nll_init_h[i] = layer.nll_init_h
            wnll_init_h[i] = layer.wnll_init_h
            
            
        self.kl  = kl
        self.wkl = wkl
        
        self.nll_init_h = nll_init_h
        self.wnll_init_h = wnll_init_h

    def prior_step(self, x=None):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.prior_step(x) if self.input else layer.prior_step()
            else:
                layer.prior_step(self.layers[i - 1].z)

    def er_initialize(self, window, situation):
        for layer in self.layers:
            layer.er_initialize(window, situation)

    def er_slide_window(self):
        for layer in self.layers:
            layer.er_slide_window()

    def er_keep_last_state(self):
        for layer in self.layers:
            layer.er_keep_last_state()

    def er_reset(self):
        for layer in self.layers:
            layer.er_reset()

    def er_step(self, x=None, itr=1):
        kl = torch.zeros(self.n_layer, device=self.device)
        wkl = torch.zeros(self.n_layer, device=self.device)
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.er_step(x, itr) if self.input else layer.er_step()
            else: #this else case includes only bottom layer of top-pvrnn (associative module)
                layer.er_step(self.layers[i - 1].z, itr)
            kl[i]  = layer.kl
            wkl[i] = layer.wkl
        self.kl  = kl
        self.wkl = wkl

    def er_prior_step(self, x=None):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.er_prior_step(x) if self.input else layer.er_prior_step()
            else:
                layer.er_prior_step(self.layers[i - 1].z) #self.layers[i - 1].d

    # change the value of w and w1 in er
    def set_w(self, w: list, w1: list):
        for _w, _w1, layer in zip(w, w1, self.layers):
            layer.w = _w
            layer.w1 = _w1
