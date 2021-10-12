"""class to facilitate to implement error regression"""

import numpy as np
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import time
import torch.optim as optim
import torch.nn as nn
np.set_printoptions(precision=5)


class ErrorRegression():
    def __init__(self, window: int, iteration: int, model, param_path, prop_target, vision_target, step: int, er_mode="prop_vision", lr=0.09,
                 first_itr=50, seed=1, situation=1):
        """
        :param window: the size of the window for error regression
        :param iteration: # iteration per time step
        :param model: model class assumed to be an integrated network
        :param param_path: a path to a trained parameter for the model
        :param prop_target: a target for proprioception in tensor
        :param vision_target: a target for vision in tensor the size of which is 64 x 64
        :param step: # time step to perform ER
        :param er_mode: "prop_vision", "prop", or "vision"
        :param lr: learning rate during ER
        :param first_itr: # iteration at t=1
        :param seed: random number seed
        """
        torch.manual_seed(seed)
        self.window  = window
        self.itr     = iteration
        self.model   = model
        self.model.load_param(param_path) # load the trained parameters
        self.prop_target   = prop_target
        self.vision_target = vision_target
        self.step = step
        self.lr   = lr
        self.first_itr = first_itr
        self.er_mode = er_mode
        self.situation = situation
        self.ps_target = torch.zeros(self.window, self.model.motor_dim, device="cpu")
        self.vs_target = torch.zeros(self.window, self.model.vision_dim, device="cpu")

        print("Initializing the model for error regression...")
        self.model.er_initialize(window, er_mode, self.situation) # initialize the model to start ER


    # compute MSE which is not averaged over time steps, but divided by the data dimension
    def compute_prop_loss(self, x, target):
        return nn.functional.mse_loss(x, target, reduction="sum") / self.model.motor_dim

    # compute MSE which is not averaged over time steps, but divided by the data dimension
    def compute_vision_loss(self, x, target):
        return nn.functional.mse_loss(x, target, reduction="sum") / self.model.vision_dim

    # compute time step-wise reconstruction loss for analysis
    def compute_timestep_prop_loss(self, x, target):
        loss = torch.zeros(x.size()[0], device="cpu")
        for i, (_x, _target) in enumerate(zip(x, target)):
            loss[i] = nn.functional.mse_loss(_x, _target, reduction="sum") / self.model.motor_dim
        return loss

    # compute time step-wise reconstruction loss for analysis
    def compute_timestep_vision_loss(self, x, target):
        loss = torch.zeros(x.size()[0], device="cpu")
        for i, (_x, _target) in enumerate(zip(x, target)):
            loss[i] = nn.functional.mse_loss(_x, _target, reduction="sum") / self.model.vision_dim
        return loss
    
    # forward kinematics,  "vs" is used to set shape of output
    def fk(self, ms, vs):
        l1, l2, l3 = [0.1, 0.3, 0.5] # lengths of links
        th = (ms + 0.8) * torch.pi / 1.6
        tht1 = th[:, 0]
        tht2 = th[:, 1]
        tht3 = th[:, 2]#joint angles
        arm_pos = torch.zeros_like(vs)

        #position of link 1
        x1 = - (l1 * torch.cos(tht1))
        y1 = l1 * torch.sin(tht1)
        
        #position of link 2
        x2 = x1 - (l2 * torch.cos((tht1+tht2)))
        y2 = y1 + (l2 * torch.sin((tht1+tht2)))
        
        #position of link 3
        x3 = x2 - (l3 * torch.cos((tht1+tht2+tht3)))
        y3 = y2 + (l3 * torch.sin((tht1+tht2+tht3)))
        
        arm_pos[:, 0] = x3
        arm_pos[:, 1] = y3
        
        #return hand position
        return arm_pos
    
    # ER at t=1
    def initial_regression(self, fn):
        """
        :param fn: file name of the results
        """
        print("Initial regression...")
        start = time.time()
        opt = optim.Adam(self.model.er_register_params(), lr=self.lr) # add "a" terms for optimization
        
        for i in range(self.first_itr):
            itr_start = time.time()

            if self.er_mode == "prop_vision":
                ps, ms, vs, top_kl, prop_kl, vision_kl, top_wkl, prop_wkl, vision_wkl = self.model.er_rnn_forward(i) #set iteration to 1 to avoid reseting last state of A
                
                # set target data according to context
                if i == 0:
                    self.ps_target = ms.clone().detach()
                    if self.situation == 0: 
                        #vision caused by other's movement
                        self.vs_target = self.vision_target[0:self.window].clone().detach()
                    if self.situation == 1:
                        #vision caused by self movement
                        self.vs_target = self.fk(self.ps_target.clone().detach(), vs.clone().detach())
                """Here is a tricky part to implement ER in PyTorch as of ver 1.4.
                It is necessary to create a new computational graph in every iteration to perform ER, but
                if once optimization is performed, PyTorch doesn't create a new computational graph over the used tensors, so to speak.
                Therefore, one needs a special treatment to go around this, and one way is to use .detach() for the target tensor as follows.
                Without this, PyTorch gives an error in next iteration"""

                # detach() the target tensor to let PyTorch create a new computational graph in next iteration

                prop_loss = self.compute_prop_loss(ps, self.ps_target.clone().detach())
                vision_loss = self.compute_vision_loss(vs, self.vs_target.clone().detach())
                loss = prop_loss + vision_loss + torch.sum(top_wkl) + torch.sum(prop_wkl) + torch.sum(vision_wkl) # compute the cost to minimize
                print("Itr {} prec {:.2f} vrec {:.2f} topkl {} pkl {} vkl {} sec/itr {:.2f} sec".format(
                    i+1, prop_loss, vision_loss, top_kl.data.numpy(), prop_kl.data.numpy(), vision_kl.data.numpy(), time.time() - itr_start))
                
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.t = 0 # set the time step inside the window zero
        print("predicting...")
        res = self.model.er_predict_forward(self.step) # compute the reconstruction after error regression and the prediction after the window
        res["prop_target"] = self.ps_target.clone().detach()
        res["vision_target"] = self.vs_target.clone().detach()
        res["hand_pos"] = self.fk(res["prop_target"][:self.window].clone().detach(), res["v"][:self.window]).clone().detach()
        res["er_p"] = self.compute_timestep_prop_loss(res["p"][:self.window], res["prop_target"][:self.window].clone().detach())
        res["er_v"] = self.compute_timestep_vision_loss(res["v"][:self.window], res["vision_target"][:self.window].clone().detach())
        torch.save(res, fn + "t_" + "0".zfill(8)) # save the result of ER at t=1
        print("Initial regression {:.2f} sec".format(time.time() - start))

    # one time step ER
    def error_regression_step(self, fn):
        step_start = time.time()
        self.model.er_slide_window()
        self.t += 1
        opt = optim.Adam(self.model.er_register_params(), lr=self.lr)
        shift_time = 100 # time step at which the situation is changed

        for i in range(self.itr):
            itr_start = time.time()
            if self.er_mode == "prop_vision":
                ps, ms, vs, top_kl, prop_kl, vision_kl, top_wkl, prop_wkl, vision_wkl = self.model.er_rnn_forward(i) # if i=0, the last state of posterior is initialized by prior
                
                # set target data at each time step according to swich pattern of environment
                if self.t + self.window - 1 >= shift_time: #Network current step = self.t + self.window - 1
                    if i == 0:     
                        self.ps_target = torch.cat((self.ps_target[1:].clone().detach(), torch.reshape(ms[-1, :].clone().detach(),(1,self.model.motor_dim))))
                        if self.situation == 0:
                            #vision caused by self movement
                            v_fk = self.fk(self.ps_target.clone().detach(), vs.clone().detach())
                            self.vs_target = torch.cat((self.vs_target[1:].clone().detach(), torch.reshape(v_fk[-1, :].clone().detach(),(1,self.model.vision_dim))))
                        if self.situation == 1:
                            #vision caused by other's movement
                            self.vs_target = torch.cat((self.vs_target[1:].clone().detach(), torch.reshape(self.vision_target[self.t + self.window - 1].clone().detach(),(1,self.model.vision_dim))))
                        
                else:
                    
                    if i == 0:
                        self.ps_target = torch.cat((self.ps_target[1:].clone().detach(), torch.reshape(ms[-1, :].clone().detach(),(1,self.model.motor_dim))))
                        if self.situation == 0:
                            #vision caused by other's movement
                            self.vs_target = torch.cat((self.vs_target[1:].clone().detach(), torch.reshape(self.vision_target[self.t + self.window - 1].clone().detach(),(1,self.model.vision_dim))))
                        if self.situation == 1:
                            #vision caused by self movement
                            v_fk = self.fk(self.ps_target.clone().detach(), vs.clone().detach())
                            self.vs_target = torch.cat((self.vs_target[1:].clone().detach(), torch.reshape(v_fk[-1, :].clone().detach(),(1,self.model.vision_dim))))
                        
                        
                prop_loss = self.compute_prop_loss(ps, self.ps_target.clone().detach())
                vision_loss = self.compute_vision_loss(vs, self.vs_target.clone().detach())
                loss = prop_loss + vision_loss + torch.sum(top_wkl) + torch.sum(prop_wkl) + torch.sum(vision_wkl)
                print(
                    "Itr {} prec {:.2f} lrec {:.2f} topkl {} pkl {} vkl {} sec/itr {:.2f} sec".format(i + 1, prop_loss, vision_loss, top_kl.data.numpy(),
                        prop_kl.data.numpy(), vision_kl.data.numpy(), time.time() - itr_start))

            opt.zero_grad()
            loss.backward()
            opt.step()
        print("predicting...")
        pred = self.model.er_predict_forward(self.step - self.t)
        pred["prop_target"] = self.ps_target.clone().detach()
        pred["vision_target"] = self.vs_target.clone().detach()
        pred["hand_pos"] = self.fk(pred["prop_target"][:self.window].clone().detach(), pred["v"][:self.window]).clone().detach()
        pred["er_p"] = self.compute_timestep_prop_loss(pred["p"][:self.window], pred["prop_target"][:self.window].clone().detach())
        pred["er_v"] = self.compute_timestep_vision_loss(pred["v"][:self.window], pred["vision_target"][:self.window].clone().detach())

        print("sec/step {:.2f}".format(time.time() - step_start))
        torch.save(pred, fn)


    # perform error regression over time steps
    def error_regression(self, path):
        start = time.time()
        self.initial_regression(path)
        for i in range(self.step - 1): #
            print("time step {}".format(i + 2))
            self.error_regression_step(path + "t_{:0>8}".format(i + 1))
        print("Total time {:.2f} min".format((time.time() - start) / 60))



