import torch
import torch.nn as nn
from model.ResBlock import ResBlock


class ResidualEncoder(nn.Module):
    def __init__(self, num_channels, depth, hidden_dim):
        super(ResidualEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([ResBlock(num_channels * (2**i)) for i in range(depth)])
        self.pool_layers = nn.ModuleList([nn.Conv2d(num_channels * (2**i),num_channels * (2**(i+1)), kernel_size=2, stride=2) for i in range(depth)])
        self.conv_f = nn.Conv2d(num_channels * (2**depth), hidden_dim, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x).relu()
        for res_block, pool_layer in zip(self.res_blocks, self.pool_layers):
            x = res_block(x).relu()
            x = pool_layer(x)
        x = self.conv_f(x)
        return x