import torch
import torch.nn as nn
from model.ResBlock import ResBlock

class ResidualDecoder(nn.Module):
    def __init__(self, num_channels, depth, hidden_dim):
        super(ResidualDecoder, self).__init__()
        self.conv1 = nn.Conv2d(hidden_dim, num_channels * (2 ** depth), kernel_size=1)
        self.res_blocks = nn.ModuleList([ResBlock(num_channels * 2**(depth-i)) for i in range(depth)])
        self.deconv_layers = nn.ModuleList([nn.ConvTranspose2d(num_channels * (2**(depth-i)),num_channels * (2**(depth-i-1)), kernel_size=2, stride=2) for i in range(depth)])
        self.conv_f = nn.Conv2d(num_channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x).relu()
        for res_block,deconv_layer in zip(self.res_blocks, self.deconv_layers):
            x = res_block(x).relu()
            x = deconv_layer(x)
        x = self.conv_f(x)
        return x