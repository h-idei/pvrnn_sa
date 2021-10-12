"output layer"

import torch.nn as nn


class Output(nn.Module):
    def __init__(self, input_size, output_size, act_func="sigmoid", device="cpu"):
        super(Output, self).__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.device      = device

        if act_func == "sigmoid": self.act_func = nn.Sigmoid()
        elif act_func == "tanh": self.act_func = nn.Tanh()
        if act_func == "no_act":
            self.layer = nn.Linear(input_size, output_size, bias=False)
        else:
            self.layer = nn.Sequential(nn.Linear(input_size, output_size, bias=False), self.act_func)

    def forward(self, d):
        x = self.layer(d)
        return x
