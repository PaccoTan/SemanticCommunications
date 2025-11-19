import torch
from torch import nn
import math

class AWGN(nn.Module):
    def __init__(self, std: float | tuple[float,float]| None):
        super(AWGN, self).__init__()
        # Define std for complex gaussian noise
        if type(std) == tuple:
            self.std_I = std[0]
            self.std_Q = std[1]
        elif type(std) == float:
            self.std_I = std*math.sqrt(0.5)
            self.std_Q = std*math.sqrt(0.5)
        else:
            self.std_I = math.sqrt(0.5)
            self.std_Q = math.sqrt(0.5)

    def forward(self, x):
        if torch.is_complex(x):
            noise_I, noise_Q = torch.normal(0, std=self.std_I, size=x.size()), torch.normal(0,std=self.std_Q, size=x.size())
            noise = torch.complex(noise_I, noise_Q)
            x = x + noise
        else:
            noise_I, noise_Q = torch.normal(0, std=self.std_I, size=x.size()[:-1]),torch.normal(0, std=self.std_Q, size=x.size()[:-1])
            noise = torch.stack([noise_I, noise_Q], dim=-1)
            x = x + noise
        return x