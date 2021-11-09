#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pylab
import re
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import torch

from optparse import OptionParser
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
params = {"backend": "pdf",
          "font.family": "Helvetica",
          "axes.titlesize": 20,
          "axes.labelsize": 20,
          "font.size": 20,
          "legend.fontsize":20,
          "xtick.labelsize": 20,
          "ytick.labelsize": 20,
          "text.usetex": False,
          "savefig.facecolor": "1.0"}
pylab.rcParams.update(params)

FOLDERNAME_GENERATE = "./result_ER/switch_StoOtoS/seed0010/lr0.1w15ite10" #
SEQUENCE_INDEX = "0"
EPOCH = "0"
SLICE = 1
T_MIN = 0
T_MAX = 200
LENGTH = 200
PHASE = 0
C_MAP = plt.cm.seismic
SEQ_NUM = 8
out_size = 5
prorprio_dim = 3
extero_dim = 2
ass_z_dim = 3
exe_z_dim = 1

ass_d_dim = 15
sensory_d_dim = 15

params = {"backend": "pdf",
          "font.family": "sans-serif",
          "font.sans-serif": "Arial",
          "axes.titlesize": 15,
          "axes.labelsize": 15,
          "font.size": 15,
          "legend.fontsize":15,
          "xtick.labelsize": 15,
          "ytick.labelsize": 15,
          "text.usetex": False,
          "savefig.facecolor": "1.0"}
pylab.rcParams.update(params)


def fk(ms, vs):
        l1, l2, l3 = [0.1, 0.3, 0.5] # lengths of links
        th = (ms + 0.8) * np.pi / 1.6
        tht1 = th[0]
        tht2 = th[1]
        tht3 = th[2]#joint angles
        arm_pos = np.zeros_like(vs)

        #position of link 1
        x1 = - (l1 * np.cos(tht1))
        y1 = l1 * np.sin(tht1)
        
        #position of link 2
        x2 = x1 - (l2 * np.cos((tht1+tht2)))
        y2 = y1 + (l2 * np.sin((tht1+tht2)))
        
        #position of link 3
        x3 = x2 - (l3 * np.cos((tht1+tht2+tht3)))
        y3 = y2 + (l3 * np.sin((tht1+tht2+tht3)))
        
        arm_pos[0] = x3
        arm_pos[1] = y3
        
        #return arm position
        return arm_pos

class plot_rnn(object):
    def __init__(self, filename_fig, filename_list, filename_target, foldername_lr):
        self.figure_name = filename_fig
        self.state_filename_list = filename_list
        self.state_filename_target = filename_target
        self.foldername = foldername_lr


    def add_info(self, ax, title, xlim, ylim, xlabel, ylabel):
        if title != None:
            ax.set_title(title)

        if xlim != None:
            ax.set_xlim(xlim)
            ax.set_xticks([0, 50, 100, 150, 200])
        else:
            ax.set_xticks([])

        if xlabel != None:
            ax.set_xlabel(xlabel)

        if ylim != None:
            ax.set_ylim(ylim)
            ax.set_yticks((ylim[0], (ylim[0] + ylim[1]) / 2.0, ylim[1]))
        if ylabel != None:
            ax.set_ylabel(ylabel)
        ax.grid(True) 
        
    def set_no_yticks(self, ax):
        ax.set_yticks([])

    def configure(self, fig_matrix, width, height):
        fig = plt.figure(figsize = (width * fig_matrix[1], height * fig_matrix[0]))
        gs = gridspec.GridSpec(fig_matrix[0], fig_matrix[1])
        axes = [fig.add_subplot(gs[i, j]) for i in range(fig_matrix[0]) for j in range(fig_matrix[1])]
        return fig, gs, axes


    def state(self, slice_num, tmin, tmax, width, height):
        fig_matrix = [7, 1]
        fig, gs, axes = self.configure(fig_matrix, width, height)

        #define varibles and read data
        in_state = np.loadtxt(self.state_filename_target)
        extero_input = in_state[:,3:]
        proprio_input = in_state[:,:3]
        target_state = np.zeros((LENGTH, 4))
        target_state_5dim = np.zeros((LENGTH, 5))
        out_state = np.zeros((LENGTH, 4))
        prediction_5dim = np.zeros((LENGTH, 5))
        out_proprio = np.zeros((LENGTH, 2))
        predicted_proprio = np.zeros((LENGTH, prorprio_dim))
        out_extero = np.zeros((LENGTH, extero_dim))
        
        exe_mu_q = np.zeros((LENGTH, exe_z_dim))
        exe_sigma_q = np.zeros((LENGTH, exe_z_dim))
        ass_d = np.zeros((LENGTH, ass_d_dim))
        ass_mu_q = np.zeros((LENGTH, ass_z_dim))
        ass_sigma_q = np.zeros((LENGTH, ass_z_dim))
        ass_mu_p = np.zeros((LENGTH, ass_z_dim))
        ass_sigma_p = np.zeros((LENGTH, ass_z_dim))
        
        proprio_d = np.zeros((LENGTH, sensory_d_dim))
        proprio_mu_q = np.zeros((LENGTH, 1))
        proprio_sigma_q = np.zeros((LENGTH, 1))
        proprio_mu_p = np.zeros((LENGTH, 1))
        proprio_sigma_p = np.zeros((LENGTH, 1))
        er_proprio = np.zeros((LENGTH, 1))
        
        extero_d = np.zeros((LENGTH, sensory_d_dim))
        extero_mu_q = np.zeros((LENGTH, 1))
        extero_sigma_q = np.zeros((LENGTH, 1))
        extero_mu_p = np.zeros((LENGTH, 1))
        extero_sigma_p = np.zeros((LENGTH, 1))
        er_extero = np.zeros((LENGTH, 1))
        
        for i, filename in enumerate(self.state_filename_list):
            data = torch.load(filename)
            #{"p": ps, "l": ls, "top_kl": top_kl, "prop_kl": prop_kl, "vision_kl": vision_kl, "top_d": top_d, "top_mu_p": top_mu_p, "top_mu_q": top_mu_q, "top_sigma_p": top_sigma_p, "top_sigma_q": top_sigma_q, "prop_d": prop_d, "prop_mu_p": prop_mu_p, "prop_mu_q": prop_mu_q, "prop_sigma_p": prop_sigma_p,"prop_sigma_q": prop_sigma_q,  "vision_d": vision_d, "vision_mu_p": vision_mu_p, "vision_mu_q": vision_mu_q, "vision_sigma_p": vision_sigma_p, "vision_sigma_q": vision_sigma_q}
            
            extero_input[i] = data["vision_target"][0].detach().numpy()
            proprio_input[i] = data["prop_target"][0].detach().numpy()
            
            target_arm = fk(data["prop_target"][0].detach().numpy(), extero_input[i, :])
            target_state[i] = np.concatenate([target_arm, extero_input[i]])
            target_state_5dim[i] = np.concatenate([proprio_input[i], extero_input[i]])
            
            out_proprio[i] = fk(data["p"][0].detach().numpy(), extero_input[i, :])
            predicted_proprio[i] = data["p"][0].detach().numpy()

            out_extero[i] = data["v"][0].detach().numpy()
            out_state[i] = np.concatenate([out_proprio[i], out_extero[i]])
            prediction_5dim[i] = np.concatenate([predicted_proprio[i], out_extero[i]])
            exe_mu_q[i] = data["top_mu_q"][0][0].detach().numpy()
            exe_sigma_q[i] = data["top_sigma_q"][0][0].detach().numpy()
            ass_d[i] = data["top_d"][1][0].detach().numpy()
            ass_mu_q[i] = data["top_mu_q"][1][0].detach().numpy()
            ass_sigma_q[i] = data["top_sigma_q"][1][0].detach().numpy()
            ass_mu_p[i] = data["top_mu_p"][1][0].detach().numpy()
            ass_sigma_p[i] = data["top_sigma_p"][1][0].detach().numpy()
            proprio_d[i] = data["prop_d"][0][0].detach().numpy()
            proprio_mu_q[i] = data["prop_mu_q"][0][0].detach().numpy()
            proprio_sigma_q[i] = data["prop_sigma_q"][0][0].detach().numpy()
            proprio_mu_p[i] = data["prop_mu_p"][0][0].detach().numpy()
            proprio_sigma_p[i] = data["prop_sigma_p"][0][0].detach().numpy()
            er_proprio[i] = data["er_p"][0].detach().numpy()
        
            extero_d[i] = data["vision_d"][0][0].detach().numpy()
            extero_mu_q[i] = data["vision_mu_q"][0][0].detach().numpy()
            extero_sigma_q[i] = data["vision_sigma_q"][0][0].detach().numpy()
            extero_mu_p[i] = data["vision_mu_p"][0][0].detach().numpy()
            extero_sigma_p[i] = data["vision_sigma_p"][0][0].detach().numpy()
            er_extero[i] = data["er_v"][0].detach().numpy()
        
        for i in range(np.array(self.state_filename_list).shape[0], LENGTH):
            
            data = torch.load(self.state_filename_list[-1]) 
            k = i - np.array(self.state_filename_list).shape[0]
            extero_input[i] = data["vision_target"][k].detach().numpy()
            proprio_input[i] = data["prop_target"][k].detach().numpy()

            target_arm = fk(data["prop_target"][k].detach().numpy(), extero_input[i, :])
            target_state[i] = np.concatenate([target_arm, extero_input[i]])
            target_state_5dim[i] = np.concatenate([proprio_input[i], extero_input[i]])
            
            out_proprio[i] = fk(data["p"][k].detach().numpy(), extero_input[i, :]) 
            predicted_proprio[i] = data["p"][k].detach().numpy()
            
            out_extero[i] = data["v"][k].detach().numpy()
            out_state[i] = np.concatenate([out_proprio[i], out_extero[i]])
            prediction_5dim[i] = np.concatenate([predicted_proprio[i], out_extero[i]])
            exe_mu_q[i] = data["top_mu_q"][0][k].detach().numpy()
            exe_sigma_q[i] = data["top_sigma_q"][0][k].detach().numpy()
            ass_d[i] = data["top_d"][1][k].detach().numpy()
            ass_mu_q[i] = data["top_mu_q"][1][k].detach().numpy()
            ass_sigma_q[i] = data["top_sigma_q"][1][k].detach().numpy()
            ass_mu_p[i] = data["top_mu_p"][1][k].detach().numpy()
            ass_sigma_p[i] = data["top_sigma_p"][1][k].detach().numpy()
            proprio_d[i] = data["prop_d"][0][k].detach().numpy()
            proprio_mu_q[i] = data["prop_mu_q"][0][k].detach().numpy()
            proprio_sigma_q[i] = data["prop_sigma_q"][0][k].detach().numpy()
            proprio_mu_p[i] = data["prop_mu_p"][0][k].detach().numpy()
            proprio_sigma_p[i] = data["prop_sigma_p"][0][k].detach().numpy()
            er_proprio[i] = data["er_p"][k].detach().numpy()
        
            extero_d[i] = data["vision_d"][0][k].detach().numpy()
            extero_mu_q[i] = data["vision_mu_q"][0][k].detach().numpy()
            extero_sigma_q[i] = data["vision_sigma_q"][0][k].detach().numpy()
            extero_mu_p[i] = data["vision_mu_p"][0][k].detach().numpy()
            extero_sigma_p[i] = data["vision_sigma_p"][0][k].detach().numpy()
            er_extero[i] = data["er_v"][k].detach().numpy()

        
        #Executive area
        axes[0].plot(exe_mu_q, linestyle="solid", linewidth="1")
        self.add_info(axes[0], None, None, (-1.2,1.2), None, "Exe_mu")
        
        #Association area
        axes[1].plot(ass_mu_q[:, 0], linestyle="solid", linewidth="1", color = "c")
        axes[1].plot(ass_mu_q[:, 1], linestyle="solid", linewidth="1", color = "m")
        axes[1].plot(ass_mu_q[:, 2], linestyle="solid", linewidth="1", color = "y")
        axes[1].plot(ass_mu_p[:, 0], linestyle="dashed", linewidth="1", color = "c")
        axes[1].plot(ass_mu_p[:, 1], linestyle="dashed", linewidth="1", color = "m")
        axes[1].plot(ass_mu_p[:, 2], linestyle="dashed", linewidth="1", color = "y")
        self.add_info(axes[1], None, None, (-1.2,1.2), None, "Ass_mu")
        
        axes[2].plot(ass_sigma_q[:, 0], linestyle="solid", linewidth="1", color = "c")
        axes[2].plot(ass_sigma_q[:, 1], linestyle="solid", linewidth="1", color = "m")
        axes[2].plot(ass_sigma_q[:, 2], linestyle="solid", linewidth="1", color = "y")
        axes[2].plot(ass_sigma_p[:, 0], linestyle="dashed", linewidth="1", color = "c")
        axes[2].plot(ass_sigma_p[:, 1], linestyle="dashed", linewidth="1", color = "m")
        axes[2].plot(ass_sigma_p[:, 2], linestyle="dashed", linewidth="1", color = "y")
        self.add_info(axes[2], None, None, (0.0,2.0), None, "Ass_sig")


        #Sensory areas
        axes[3].plot(extero_mu_q, linestyle="solid", linewidth="1", color="r")
        axes[3].plot(proprio_mu_q, linestyle="solid", linewidth="1", color="b")
        axes[3].plot(extero_mu_p, linestyle="dashed", linewidth="1", color="r")
        axes[3].plot(proprio_mu_p, linestyle="dashed", linewidth="1", color="b")
        self.add_info(axes[3], None, None, (-1.2,1.2), None, "Sen_mu")
        
        axes[4].plot(extero_sigma_q, linestyle="solid", linewidth="1", color="r")
        axes[4].plot(proprio_sigma_q, linestyle="solid", linewidth="1", color="b")
        axes[4].plot(extero_sigma_p, linestyle="dashed", linewidth="1", color="r")
        axes[4].plot(proprio_sigma_p, linestyle="dashed", linewidth="1", color="b")
        self.add_info(axes[4], None, None, (0.0,2.0), None, "Sen_sig")
       
        #prediction error
        axes[5].plot(er_extero, linestyle="solid", linewidth="1", color="r")
        axes[5].plot(er_proprio, linestyle="solid", linewidth="1", color="b")
        self.add_info(axes[5], None, None, (0.0, 0.3), None, "PE")
        
        #Real sensations
        axes[6].plot(target_state[:, 2:], linestyle="solid", linewidth="1", color="r") #extero
        axes[6].plot(target_state[:, :2], linestyle="solid", linewidth="1", color="b") #prop
        self.add_info(axes[6], None, (0,200), (-1.2,1.2), "Time", "Real")
        
        
        for ax in axes:
            ax.set_xlim(tmin, tmax)
        

        fig.savefig(self.figure_name, format="pdf",dpi=300)
        #sequence = {"target": target_state_5dim, "prediction": prediction_5dim, "top_mu_q": exe_mu_q, "top_sigma_q": exe_sigma_q, "ass_mu_q": ass_mu_q, "ass_sigma_q": ass_sigma_q, "ass_mu_p": ass_mu_p, "ass_sigma_p": ass_sigma_p, "prop_mu_q": proprio_mu_q, "prop_sigma_q": proprio_sigma_q, "prop_mu_p": proprio_mu_p, "prop_sigma_p": proprio_sigma_p, "vision_mu_q": extero_mu_q, "vision_sigma_q": extero_sigma_q, "vision_mu_p": extero_mu_p, "vision_sigma_p": extero_sigma_p}
        #np.save(self.foldername + "result", sequence)


def main():
    
    foldername_list_sit = sorted(glob.glob("./result_ER/sit*/"))
    print(foldername_list_sit)
    for c, foldername_sit in enumerate(foldername_list_sit):
        foldername_list_seq = sorted(glob.glob(foldername_sit + "seq*/"))
        print(foldername_list_seq)
        for s, foldername_seq in enumerate(foldername_list_seq):
            foldername_list_ite = sorted(glob.glob(foldername_seq + "ite*/"))
            print(foldername_list_ite)
            for k, foldername_ite in enumerate(foldername_list_ite):
                foldername_list_window = sorted(glob.glob(foldername_ite + "window*/"))
                for i, foldername_window in enumerate(foldername_list_window):
                    foldername_list_lr = sorted(glob.glob(foldername_window + "lr*/"))
                    for j, foldername_lr in enumerate(foldername_list_lr):
                        filename_list = sorted(glob.glob(foldername_lr + "t_*"))
                        filename_fig = foldername_lr + "/generate_journal.pdf"
                        filename_target = "./target/target_0000.txt" #target_0000.txt is just used for set the shape of array of sensations
                        plot = plot_rnn(filename_fig, filename_list, filename_target, foldername_lr)
                        plot.state(SLICE, T_MIN, T_MAX, 7, 2)
        

if __name__ == "__main__":
    main()
