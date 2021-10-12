"""An example of training a network"""

import matplotlib
matplotlib.use("Agg")
import os
import glob
import sys
sys.path.append("../")
from networks.integrated_network import Integrated
from dataset import DataSet
from utilities import *
import torch
import torch.optim as optim
import time
import numpy as np
import yaml
np.set_printoptions(precision=2)
torch.pi = torch.acos(torch.zeros(1)) * 2 # which is 3.1415927410125732
def main():

    result_path = "./result_training/"
    seed = 2          # random number seed: 1-10
    minibatch_size = 48 # minibatch size

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_np  = np.random.rand(48, 200, 5)
    motor_train_np  = np.random.rand(48, 200, 3)
    vision_train_np  = np.random.rand(48, 200, 2)
    directory_list = sorted(glob.glob("./target/*.txt"))
    for i, directory in enumerate(directory_list):
        train_np[i] = np.loadtxt(directory)
    
    motor_train_np  = train_np[:,:,:3]
    vision_train_np = train_np[:,:,3:]
    dataset = DataSet(motor_train_np, vision_train_np, minibatch_size)
    epo = 200000 # epochs for training
    lr = 1e-3   # learning rate for Adam optimizer

    """In this example, a network configuration is loaded with yaml.
    Whatever way is fine as long as a configuration is properly passed to the network as a dictionary"""

    with open("network_config.yaml") as f:
        param = yaml.load(f)

    print("network configuration: {}".format(param))

    # load model configuration and dataset
    model = Integrated(param, minibatch_size, dataset.n_minibatch, dataset.seq_len, dataset.motor_dim, dataset.vision_dim)

    # assign rnn parameters for updating and set a learning rate
    opt   = optim.Adam([{"params": model.top.parameters()},
                        {"params": model.prop.parameters()},
                        {"params": model.vision.parameters()},
                        {"params": model.prop_out.parameters()},
                        {"params": model.vision_out.parameters()}],lr=lr)

    # compute reconstruction error
    prop_mse = NormalizeMSE(dataset.motor_dim)
    vision_mse = NormalizeMSE(dataset.vision_dim)

    start = time.time()
    for i in range(epo):
        epo_start = time.time()
        model.train()

        # initialize each loss per epoch
        _top_kl, _prop_kl, _vision_kl, _kl = 0., 0., 0., 0.
        _top_nll_init_h, _prop_nll_init_h, _vision_nll_init_h, _nll_init_h = 0., 0., 0., 0.
        _p_loss, _v_loss, _loss = 0., 0., 0.

        for j in np.random.permutation(range(dataset.n_minibatch)):
            ps, vs, top_kl, prop_kl, vision_kl, top_wkl, prop_wkl, vision_wkl, top_wnll_init_h, prop_wnll_init_h, vision_wnll_init_h, res = model.posterior_forward(i, j)
            p_loss = prop_mse(ps, dataset.motor_minibatch[j])
            v_loss = vision_mse(vs, dataset.vision_minibatch[j])
            
            
            loss = p_loss + v_loss + torch.sum(top_wkl) + torch.sum(prop_wkl) + torch.sum(vision_wkl) + torch.sum(top_wnll_init_h) + torch.sum(prop_wnll_init_h) + torch.sum(vision_wnll_init_h)
            
            
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            _top_kl += top_kl
            _prop_kl += prop_kl
            _vision_kl += vision_kl
            _kl += torch.sum(top_kl) + torch.sum(prop_kl) + torch.sum(vision_kl)
            _p_loss += p_loss.item()
            _v_loss += v_loss.item()
            _loss   += loss.item()

            print("minibatch_ind {}/{} loss {:.3f}".format(j+1, dataset.n_minibatch, loss))

        _top_kl    /= dataset.n_minibatch
        _prop_kl   /= dataset.n_minibatch
        _vision_kl /= dataset.n_minibatch
        _kl        /= dataset.n_minibatch
        _p_loss    /= dataset.n_minibatch
        _v_loss    /= dataset.n_minibatch
        _loss      /= dataset.n_minibatch
        
        res["kl"] = _kl
        res["mse"] = _p_loss + _v_loss
        res["loss"] = _loss
        if i % 5000 == 0:
                model.save_param("./trained_model/model_{:0>8}".format(i) + ".pth")
                np.save(result_path + "generate_{:0>8}".format(i) + ".npy", res)

        print("Epo {} loss {:.5f} epoch/sec {:.2f} total {:.2f} h".format(
            i+1, _loss, time.time() - epo_start, (time.time() - start) / 3600))

    print("end")
    
    print(res["top_bias_d"])
    print(res["prop_bias_d"])
    print(res["vision_bias_d"])
    
    model.save_param("./trained_model/model_{:0>8}".format(epo) + ".pth")
    np.save(result_path + "generate_{:0>8}".format(epo) + ".npy", res)


if __name__ == "__main__":
    args = sys.argv
    main()



