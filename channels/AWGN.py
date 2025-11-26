import torch
from torch import nn
import math

class AWGN(nn.Module):
    def __init__(self, std: float | tuple[float,float]| None):
        super(AWGN, self).__init__()
        # Define std for complex gaussian noise
        if type(std) == tuple:
            std_I = std[0]
            std_Q = std[1]
        elif type(std) == float:
            std_I = std*math.sqrt(0.5)
            std_Q = std*math.sqrt(0.5)
        else:
            std_I = math.sqrt(0.5)
            std_Q = math.sqrt(0.5)
        self.I_noise = torch.distributions.Normal(0, std_I)
        self.Q_noise = torch.distributions.Normal(0, std_Q)

    def forward(self, x, device="cuda"):
        if torch.is_complex(x):
            noise_I, noise_Q = self.I_noise.sample(x.size()), self.Q_noise.sample(x.size())
            noise = torch.complex(noise_I, noise_Q).to(device)
            x = x + noise
        else:
            noise_I, noise_Q = self.I_noise.sample(x.size()[:-1]),self.Q_noise.sample(x.size()[:-1])
            noise = torch.stack([noise_I, noise_Q], dim=-1).to(device)
            x = x + noise
        return x