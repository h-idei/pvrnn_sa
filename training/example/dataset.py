"""dataset class to train visuo-motor associative network"""

import torch


class DataSet():
    def __init__(self, motor, vision, minibatch_size: int):
        """motor: npy file with the form of(n_seq, time-step, n_joints)
           vision: npy file with the form of (n_seq, time-step, height, width)
           minibatch_size: should hold n_seq%minibatch_size=0"""

        self.minibatch_size = minibatch_size
        self.n_seq, self.seq_len, self.motor_dim = motor.shape
        #self.vision_dim = vision.shape[-2] * vision.shape[-1]
        self.vision_dim = vision.shape[-1]
        if self.n_seq % minibatch_size != 0:
            print("Choose minibatch_size s.t. n_seq % minibatch_size = 0")

        self.n_minibatch = int(self.n_seq / minibatch_size)
        print("#seq: {}, minibatch_size: {}, #minibatch: {}".format(self.n_seq, minibatch_size, self.n_minibatch))

        motor_minibatch, vision_minibatch = [], []
        for i in range(self.n_minibatch):
            motor_minibatch.append(torch.from_numpy(motor[i * minibatch_size: (i + 1) * minibatch_size]).type(torch.FloatTensor))
            #vision_minibatch.append(torch.from_numpy(vision[i * minibatch_size: (i + 1) * minibatch_size]).type(torch.FloatTensor).to("cuda"))
            vision_minibatch.append(torch.from_numpy(vision[i * minibatch_size: (i + 1) * minibatch_size]).type(torch.FloatTensor))
    
        self.motor_minibatch = motor_minibatch
        self.vision_minibatch = vision_minibatch


class VisionData():
    def __init__(self, data, minibatch_size: int):
        """data: npy file with the form of (n_seq, time-step, height, width)
            minibach_size: should hole n_seq % minibatch_size = 0"""
        self.minibatch_size = minibatch_size
        self.n_seq, self.seq_len, _, _ = data.shape
        if self.n_seq % minibatch_size != 0:
            print("#seq: {}, minibatch_size: {} - Choose minibatch_size s.t. #seq % minibatch_size = 0".format(self.n_seq, minibatch_size))
        self.n_minibatch = int(self.n_seq / minibatch_size)
        print("#seq: {}, minibatch_size: {}, #minibatch: {}".format(self.n_seq, minibatch_size, self.n_minibatch))
        minibatch = []
        for i in range(self.n_minibatch):
            minibatch.append(torch.from_numpy(data[i * minibatch_size: (i + 1) * minibatch_size]).type(torch.FloatTensor).to("cuda"))
        self.minibatch = minibatch


class MotorAndLatent():
    def __init__(self, motor, latent):
        """motor: npy file
           latent: list of minibatches of torch.FloatTensors"""
        self.latent_minibatch = latent
        self.minibatch_size, self.seq_len, _ = latent[0].shape
        self.n_minibatch = len(latent)
        self.n_seq       = self.minibatch_size * self.n_minibatch
        print("#seq: {}, minibatch_size: {}, #minibatch: {}".format(self.n_seq, self.minibatch_size, self.n_minibatch))
        motor_minibatch = []
        for i in range(self.n_minibatch):
            motor_minibatch.append(torch.from_numpy(motor[i * self.minibatch_size: (i + 1) * self.minibatch_size]).type(torch.FloatTensor))
        self.motor_minibatch = motor_minibatch


