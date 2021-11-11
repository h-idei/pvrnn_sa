import numpy as np
import os.path
import sys
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
from optparse import OptionParser
import glob
Q_size_sen = 1
Q_size_ass = 3
T_MIN = 0
T_MAX = 200
SEQ_NUM = 48
FOLDERNAME_GENERATE = "./result_training"

if __name__ == '__main__':

    filename_data_list = sorted(glob.glob("./result_training/*.npy")) 
    
    prop_epoch_change_self = np.zeros((np.array(filename_data_list).shape[0], 1))
    prop_epoch_change_others = np.zeros((np.array(filename_data_list).shape[0], 1))
    extero_epoch_change_self = np.zeros((np.array(filename_data_list).shape[0], 1))
    extero_epoch_change_others = np.zeros((np.array(filename_data_list).shape[0], 1))
    
    prop_epoch_sigma_self = np.zeros((np.array(filename_data_list).shape[0], 1))
    prop_epoch_sigma_others = np.zeros((np.array(filename_data_list).shape[0], 1))
    extero_epoch_sigma_self = np.zeros((np.array(filename_data_list).shape[0], 1))
    extero_epoch_sigma_others = np.zeros((np.array(filename_data_list).shape[0], 1))
    
    for k, filename in enumerate(filename_data_list): #k: epoch
        print(filename)
        data = np.load(filename, allow_pickle=True)
        prop_epoch_change_seq = np.zeros(SEQ_NUM)
        extero_epoch_change_seq = np.zeros(SEQ_NUM)
        
        prop_epoch_sigma_seq = np.zeros(SEQ_NUM)
        extero_epoch_sigma_seq = np.zeros(SEQ_NUM)
        
        for i in range(SEQ_NUM):
            prop_q_mu = data.item().get("prop_mu_q")[0][i, :, :].detach().numpy()
            prop_p_sigma = data.item().get("prop_sigma_p")[0][i, :, :].detach().numpy()
            extero_q_mu = data.item().get("vision_mu_q")[0][i, :, :].detach().numpy()
            extero_p_sigma = data.item().get("vision_sigma_p")[0][i, :, :].detach().numpy()
            
            #If you want to get posterior or prior value in assocation or executive area
            #association_q_mu = data.item().get("top_mu_q")[1][i, :, :].detach().numpy()
            #association_p_sigma = data.item().get("top_sigma_p")[1][i, :, :].detach().numpy()
            #executive_q_mu = data.item().get("top_mu_q")[0][i, :, :].detach().numpy()
            #executive_p_sigma = data.item().get("top_sigma_p")[0][i, :, :].detach().numpy()
            #[i, :, :] = [index of target sequence, index of time step, index of neuron]
            
            #Calculation of posterior response    
            prop_change = 0
            extero_change = 0
            
            for t in range(T_MAX-1):
                prop_change += np.sqrt( np.sum ( (prop_q_mu[t+1,:] - prop_q_mu[t,:]) ** 2)) / T_MAX / Q_size_sen
                extero_change += np.sqrt( np.sum ( (extero_q_mu[t+1,:] - extero_q_mu[t,:]) ** 2)) / T_MAX / Q_size_sen
                
            prop_epoch_change_seq[i] = prop_change
            extero_epoch_change_seq[i] = extero_change

            #Calculation of level of prior sigma
            prop_epoch_sigma_seq[i] = np.average(prop_p_sigma)
            extero_epoch_sigma_seq[i] = np.average(extero_p_sigma)
                
        #average over all sequences
        prop_epoch_change_self[k] = np.average(prop_epoch_change_seq[:24])
        prop_epoch_change_others[k] = np.average(prop_epoch_change_seq[24:])
        extero_epoch_change_self[k] = np.average(extero_epoch_change_seq[:24])
        extero_epoch_change_others[k] = np.average(extero_epoch_change_seq[24:])
        
        prop_epoch_sigma_self[k] = np.average(prop_epoch_sigma_seq[:24])
        prop_epoch_sigma_others[k] = np.average(prop_epoch_sigma_seq[24:])
        extero_epoch_sigma_self[k] = np.average(extero_epoch_sigma_seq[:24])
        extero_epoch_sigma_others[k] = np.average(extero_epoch_sigma_seq[24:])
            

    fig = plt.figure(figsize=(6.4, 4.8), dpi=300, facecolor='w', linewidth=0, edgecolor='w')
    ax = fig.add_subplot(111)
    ax.plot(extero_epoch_change_self, color='r', linestyle='solid')
    ax.plot(prop_epoch_change_self, color='r', linestyle='dotted')
    ax.plot(extero_epoch_change_others, color='b', linestyle='solid')
    ax.plot(prop_epoch_change_others, color='b', linestyle='dotted')
    ax.set_xlabel("Learning epoch (*5000)")
    ax.set_ylabel("Sensory posterior response")
    ax.set_xlim((0, 40))
    ax.set_xticks((0, 10, 20, 30, 40))
    ax.set_ylim((0, 1.0))
    ax.set_yticks((0, 0.5, 1.0))
    filename_fig = "./development_sensory_posterior_response.pdf"
    fig.savefig(filename_fig, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    
    
    fig = plt.figure(figsize=(6.4, 4.8), dpi=300, facecolor='w', linewidth=0, edgecolor='w')
    ax = fig.add_subplot(111)
    ax.plot(extero_epoch_sigma_self, color='r', linestyle='solid')
    ax.plot(prop_epoch_sigma_self, color='r', linestyle='dotted')
    ax.plot(extero_epoch_sigma_others, color='b', linestyle='solid')
    ax.plot(prop_epoch_sigma_others, color='b', linestyle='dotted')
    ax.set_xlabel("Learning epoch (*5000)")
    ax.set_ylabel("Sensory prior Sigma")
    ax.set_xlim((0, 40))
    ax.set_xticks((0, 10, 20, 30, 40))
    ax.set_ylim((0, 2.0))
    ax.set_yticks((0, 1.0, 2.0))
    filename_fig = "./development_sensory_prior_sigma.pdf"
    fig.savefig(filename_fig, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    
    
