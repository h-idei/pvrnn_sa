"""
Integrated network composed of multi-layered PV-RNN dealing with vision and proprioception simultaneously which
- has vertical connection between layers
- is compatible with mini-batch learning
- samples prior from unit gaussian prior at t=0
"""
import sys
sys.path.append("../")

from networks.pvrnn import PVRNN
from networks.output import Output
import torch
import torch.nn as nn


class Integrated(nn.Module):
    def __init__(self, net_param: dict, minibatch_size: int, n_minibatch: int, seq_len: int, motor_dim=3, vision_dim=2):
        super(Integrated, self).__init__()
        print("Initializing the network...")
        self.net_param       = net_param # hyper parameters are given with a dictionary
        self.minibatch_size  = minibatch_size
        self.n_minibatch     = n_minibatch
        self.seq_len     = seq_len
        self.motor_dim   = motor_dim
        self.vision_dim  = vision_dim

        """initialize each module of the entire network.
        PV-RNN parts are loaded on CPU."""
        # assiciative module on CPU
        self.top = PVRNN(net_param["top_d_size"], net_param["top_z_size"], net_param["top_tau"], net_param["top_w"], net_param["top_w1"], net_param["top_wd1"], minibatch_size, n_minibatch, seq_len, device="cpu")
        # proprioception module on CPU
        self.prop = PVRNN(net_param["prop_d_size"], net_param["prop_z_size"], net_param["prop_tau"], net_param["prop_w"], net_param["prop_w1"], net_param["prop_wd1"], minibatch_size, n_minibatch, seq_len, input=True, input_dim=net_param["top_d_size"][-1], device="cpu")
        # proprioception output on CPU
        self.prop_out = Output(net_param["prop_d_size"][-1], motor_dim, act_func="tanh", device="cpu")
        # vision module - PV-RNN part on CPU
        self.vision = PVRNN(net_param["vision_d_size"], net_param["vision_z_size"], net_param["vision_tau"], net_param["vision_w"], net_param["vision_w1"], net_param["vision_wd1"], minibatch_size, n_minibatch, seq_len, input=True, input_dim=net_param["top_d_size"][-1], device="cpu")
        # vision module - PV-RNN part output on CPU
        self.vision_out = Output(net_param["vision_d_size"][-1], vision_dim, act_func="tanh", device="cpu")

    def initialize(self, minibatch_ind: int):
        self.top.initialize(minibatch_ind)
        self.prop.initialize(minibatch_ind)
        self.vision.initialize(minibatch_ind)

    # change the value of w and w1 in er
    def set_w(self, w_setting: dict):
        self.top.set_w(w_setting["top_w"], w_setting["top_w1"])
        self.prop.set_w(w_setting["prop_w"], w_setting["prop_w1"])
        self.vision.set_w(w_setting["vision_w"], w_setting["vision_w1"])

    # compute one time step forward computation with posterior
    def posterior_step(self, epo: int):
        self.top.posterior_step(epo, None)
        self.prop.posterior_step(epo, self.top.layers[-1].d)
        p = self.prop_out(self.prop.layers[-1].d)
        self.vision.posterior_step(epo, self.top.layers[-1].d)
        v = self.vision_out(self.vision.layers[-1].d)

        return p, v

    # compute one time step forward computation with prior
    def prior_step(self):
        self.top.prior_step()
        self.prop.prior_step(self.top.layers[-1].d)
        p = self.prop_out(self.prop.layers[-1].d)
        self.vision.prior_step(self.top.layers[-1].d)
        v = self.vision_out(self.vision.layers[-1].d)
        return p, v


    # generate latent vision with posterior (lt) and embed a pixel image (l)
    def posterior_enc_step(self, x):
        self.top.posterior_step()
        self.vision.posterior_step(self.top.layers[-1].d)
        lt = self.vision_out(self.vision.layers[-1].d)
        l  = self.encoder(x).cpu().view(self.minibatch_size, self.latent_size)
        return l, lt

    # generate proprioception and vision with posterior
    def posterior_forward(self, epo: int, minibatch_ind: int):
        ps = torch.zeros(self.minibatch_size, self.seq_len, self.motor_dim, device="cpu")
        vs = torch.zeros(self.minibatch_size, self.seq_len, self.vision_dim, device="cpu")
        self.initialize(minibatch_ind)
        
        top_kl    = torch.zeros(len(self.net_param["top_z_size"]), device="cpu")
        prop_kl   = torch.zeros(len(self.net_param["prop_z_size"]), device="cpu")
        vision_kl = torch.zeros(len(self.net_param["vision_z_size"]), device="cpu")
        top_wkl    = torch.zeros(len(self.net_param["top_z_size"]), device="cpu")
        prop_wkl   = torch.zeros(len(self.net_param["prop_z_size"]), device="cpu")
        vision_wkl = torch.zeros(len(self.net_param["vision_z_size"]), device="cpu")

        top_d, top_mu_p, top_mu_q, top_sigma_p, top_sigma_q, top_kl_step, top_a, top_init_h, top_init_h_mu, top_bias = [], [], [], [], [], [], [], [], [], []
        for l in range(self.top.n_layer):
            top_d.append(torch.zeros(self.minibatch_size, self.seq_len, self.top.d_size[l], device="cpu"))
            top_mu_p.append(torch.zeros(self.minibatch_size, self.seq_len, self.top.z_size[l], device="cpu"))
            top_mu_q.append(torch.zeros(self.minibatch_size, self.seq_len, self.top.z_size[l], device="cpu"))
            top_sigma_p.append(torch.zeros(self.minibatch_size, self.seq_len, self.top.z_size[l], device="cpu"))
            top_sigma_q.append(torch.zeros(self.minibatch_size, self.seq_len, self.top.z_size[l], device="cpu"))
            top_kl_step.append(torch.zeros(self.seq_len, device="cpu"))
            top_a.append(torch.zeros(self.minibatch_size, self.seq_len, 2*self.top.z_size[l], device="cpu"))
            top_init_h.append(torch.zeros(self.minibatch_size, self.top.d_size[l], device="cpu"))
            top_init_h_mu.append(torch.zeros(self.top.d_size[l], device="cpu"))
            top_bias.append(torch.zeros(self.top.d_size[l], device="cpu"))

        prop_d, prop_mu_p, prop_mu_q, prop_sigma_p, prop_sigma_q, prop_kl_step, prop_a, prop_init_h, prop_init_h_mu, prop_bias = [], [], [], [], [], [], [], [], [], []
        for l in range(self.prop.n_layer):
            prop_d.append(torch.zeros(self.minibatch_size, self.seq_len, self.prop.d_size[l], device="cpu"))
            prop_mu_p.append(torch.zeros(self.minibatch_size, self.seq_len, self.prop.z_size[l], device="cpu"))
            prop_mu_q.append(torch.zeros(self.minibatch_size, self.seq_len, self.prop.z_size[l], device="cpu"))
            prop_sigma_p.append(torch.zeros(self.minibatch_size, self.seq_len, self.prop.z_size[l], device="cpu"))
            prop_sigma_q.append(torch.zeros(self.minibatch_size, self.seq_len, self.prop.z_size[l], device="cpu"))
            prop_kl_step.append(torch.zeros(self.seq_len, device="cpu"))
            prop_a.append(torch.zeros(self.minibatch_size, self.seq_len, 2*self.prop.z_size[l], device="cpu"))
            prop_init_h.append(torch.zeros(self.minibatch_size, self.prop.d_size[l], device="cpu"))
            prop_init_h_mu.append(torch.zeros(self.prop.d_size[l], device="cpu"))
            prop_bias.append(torch.zeros(self.prop.d_size[l], device="cpu"))

        vision_d, vision_mu_p, vision_mu_q, vision_sigma_p, vision_sigma_q, vision_kl_step, vision_a, vision_init_h, vision_init_h_mu, vision_bias = [], [], [], [], [], [], [], [], [], []
        for l in range(self.vision.n_layer):
            vision_d.append(torch.zeros(self.minibatch_size, self.seq_len, self.vision.d_size[l], device="cpu"))
            vision_mu_p.append(torch.zeros(self.minibatch_size, self.seq_len, self.vision.z_size[l], device="cpu"))
            vision_mu_q.append(torch.zeros(self.minibatch_size, self.seq_len, self.vision.z_size[l], device="cpu"))
            vision_sigma_p.append(torch.zeros(self.minibatch_size, self.seq_len, self.vision.z_size[l], device="cpu"))
            vision_sigma_q.append(torch.zeros(self.minibatch_size, self.seq_len, self.vision.z_size[l], device="cpu"))
            vision_kl_step.append(torch.zeros(self.seq_len, device="cpu"))
            vision_a.append(torch.zeros(self.minibatch_size, self.seq_len, 2*self.vision.z_size[l], device="cpu"))
            vision_init_h.append(torch.zeros(self.minibatch_size, self.vision.d_size[l], device="cpu"))
            vision_init_h_mu.append(torch.zeros(self.vision.d_size[l], device="cpu"))
            vision_bias.append(torch.zeros(self.vision.d_size[l], device="cpu"))
        
        for t in range(self.seq_len):
            ps[:, t, :], vs[:, t, :] = self.posterior_step(epo)
            
            for l, layer in enumerate(self.top.layers):
                top_d[l][:, t, :] = layer.d
                top_mu_p[l][:, t, :] = layer.mu_p
                top_mu_q[l][:, t, :] = layer.mu_q
                top_sigma_p[l][:, t, :] = layer.sigma_p
                top_sigma_q[l][:, t, :] = layer.sigma_q
                top_kl_step[l][t] = layer.kl
                
            for l, layer in enumerate(self.prop.layers):
                prop_d[l][:, t, :] = layer.d
                prop_mu_p[l][:, t, :] = layer.mu_p
                prop_mu_q[l][:, t, :] = layer.mu_q
                prop_sigma_p[l][:, t, :] = layer.sigma_p
                prop_sigma_q[l][:, t, :] = layer.sigma_q
                prop_kl_step[l][t] = layer.kl

                
            for l, layer in enumerate(self.vision.layers):
                vision_d[l][:, t, :] = layer.d
                vision_mu_p[l][:, t, :] = layer.mu_p
                vision_mu_q[l][:, t, :] = layer.mu_q
                vision_sigma_p[l][:, t, :] = layer.sigma_p
                vision_sigma_q[l][:, t, :] = layer.sigma_q
                vision_kl_step[l][t] = layer.kl

            
            top_kl     += self.top.kl
            prop_kl    += self.prop.kl
            vision_kl  += self.vision.kl
            top_wkl    += self.top.wkl
            prop_wkl   += self.prop.wkl
            vision_wkl += self.vision.wkl
        
        top_wnll_init_h = self.top.wnll_init_h
        prop_wnll_init_h = self.prop.wnll_init_h 
        vision_wnll_init_h = self.vision.wnll_init_h
        
        for l, layer in enumerate(self.top.layers):
            top_a[l] = self.top.state_dict()["layers." + str(l) + ".A.0"]
            top_init_h[l] = self.top.state_dict()["layers." + str(l) + ".init_h.0"]
            top_init_h_mu[l] = self.top.state_dict()["layers." + str(l) + ".init_h_mu.0"]
            top_bias[l] = layer.bias_d
        
        for l, layer in enumerate(self.prop.layers):
             prop_a[l] = self.prop.state_dict()["layers." + str(l) + ".A.0"]
             prop_init_h[l] = self.prop.state_dict()["layers." + str(l) + ".init_h.0"]
             prop_init_h_mu[l] = self.prop.state_dict()["layers." + str(l) + ".init_h_mu.0"]
             prop_bias[l] = layer.bias_d
        
        for l, layer in enumerate(self.vision.layers):
             vision_a[l] = self.vision.state_dict()["layers." + str(l) + ".A.0"]
             vision_init_h[l] = self.vision.state_dict()["layers." + str(l) + ".init_h.0"]
             vision_init_h_mu[l] = self.vision.state_dict()["layers." + str(l) + ".init_h_mu.0"]
             vision_bias[l] = layer.bias_d
        
        return ps, vs, top_kl, prop_kl, vision_kl, top_wkl, prop_wkl, vision_wkl, top_wnll_init_h, prop_wnll_init_h, vision_wnll_init_h, {"p": ps, "v": vs, "top_kl": top_kl_step, "prop_kl": prop_kl_step, "vision_kl": vision_kl_step, "top_d": top_d, "top_mu_p": top_mu_p, "top_mu_q": top_mu_q, "top_sigma_p": top_sigma_p, "top_sigma_q": top_sigma_q, "prop_d": prop_d, "prop_mu_p": prop_mu_p, "prop_mu_q": prop_mu_q, "prop_sigma_p": prop_sigma_p, "prop_sigma_q": prop_sigma_q, "vision_d": vision_d, "vision_mu_p": vision_mu_p, "vision_mu_q": vision_mu_q, "vision_sigma_p": vision_sigma_p, "vision_sigma_q": vision_sigma_q, "top_a": top_a, "prop_a": prop_a, "vision_a": vision_a, "top_init_h": top_init_h, "top_init_h_mu": top_init_h_mu, "prop_init_h": prop_init_h, "prop_init_h_mu": prop_init_h_mu, "vision_init_h": vision_init_h, "vision_init_h_mu": vision_init_h_mu, "top_bias_d": top_bias, "prop_bias_d": prop_bias, "vision_bias_d": vision_bias}


    # save a trained parameter
    def save_param(self, fn="para.pth"):
        para = self.state_dict()
        torch.save(para, fn)

    # load a trained parameter
    def load_param(self, fn="para.pth"):
        param = torch.load(fn)
        self.load_state_dict(param)

