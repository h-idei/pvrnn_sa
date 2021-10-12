import os, sys
import glob
sys.path.append("../")
from networks.integrated_network import Integrated
from error_regression import ErrorRegression
from dataset import DataSet
from utilities import *
import torch
import numpy as np
import yaml
np.set_printoptions(precision=2)


def main():
    """ER specifications"""
    test_data_num = 8
    iteration_list = [50]# iterations per time step
    initial_iteration = iteration_list[0] #iteration at t=1
    window_list = [10]# window size
    lr_list = [0.09] # learning rate during ER
    time_length = 200
    total_step = time_length - window_list[0]         # total time steps to simulate
    

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    minibatch_size = 48

    # set test data
    test_np  = np.random.rand(48, 200, 5)
    motor_test_np  = np.random.rand(48, 200, 3)
    vision_test_np  = np.random.rand(48, 200, 2)
    directory_list = sorted(glob.glob("./target/*.txt"))
    for i, directory in enumerate(directory_list):
        test_np[i] = np.loadtxt(directory)
    
    motor_test_np= test_np[:,:,:3]
    vision_test_np = test_np[:,:,3:] 
    dataset = DataSet(motor_test_np, vision_test_np, minibatch_size)
    
    motor_test = torch.from_numpy(motor_test_np).type(torch.FloatTensor)
    vision_test = torch.from_numpy(vision_test_np).type(torch.FloatTensor)
    
    
    # set trained parameter
    param_path = "./model_00200000.pth" #"path to a parameter set of a trained model"
    bias = np.load("./generate_00200000.npy", allow_pickle=True).item()



    with open("network_config.yaml") as f: #"network_configuration.yaml"
        model_config = yaml.load(f)
        
    # for-loop includes those for number of test sequence, iteration, window size, and learning rate
    foldername_sit = "./result_ER/sit01/"
    c = 1	#c=1: from self-produced to other-produced
    foldername_list_seq = sorted(glob.glob(foldername_sit + "seq*/"))
    print(foldername_list_seq)
    for s, foldername_seq in enumerate(foldername_list_seq):  
        foldername_list_ite = sorted(glob.glob(foldername_seq + "ite*/"))
        print(foldername_list_ite)
        for k, foldername_ite in enumerate(foldername_list_ite):
            iteration = iteration_list[k]
            foldername_list_window = sorted(glob.glob(foldername_ite + "window*/"))
            print(foldername_list_window)
            for i, foldername_window in enumerate(foldername_list_window):
                window = window_list[i]
                total_step = time_length - window
                foldername_list_lr = sorted(glob.glob(foldername_window + "lr*/"))
                print(foldername_list_lr)
                for j, foldername_lr in enumerate(foldername_list_lr):
                    lr = lr_list[j]
                    result_path = foldername_lr
                    seed = s + 1
                    model = Integrated(model_config, bias, minibatch_size, dataset.n_minibatch, dataset.seq_len, dataset.motor_dim, dataset.vision_dim)
                    
                    # test with the s th sequence in motor_test and vision_test
                    ER = ErrorRegression(window, iteration, model, param_path, motor_test[s], vision_test[s], total_step, er_mode="prop_vision", lr=lr, first_itr=initial_iteration, seed=seed, situation = c)
                    ER.error_regression(result_path)


    print("Finished")


if __name__ == "__main__":
    main()
