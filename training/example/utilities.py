import matplotlib
matplotlib.use("Agg")

import torch.nn as nn
import seaborn as sns

sns.set_context("talk")


# compute MSE w/o averaging, divided by data dimension
class NormalizeMSE(nn.Module):
    def __init__(self, dim):
        super(NormalizeMSE, self).__init__()
        self.dim = dim

    def forward(self, x, target):
        return nn.functional.mse_loss(x, target, reduction="sum") / self.dim


