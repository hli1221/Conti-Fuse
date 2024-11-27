import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, num_L): 
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_chan, out_channels=out_chan, \
                               stride=1, kernel_size=3, padding=1, groups=num_L)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_chan, out_channels=out_chan, \
                               stride=1, kernel_size=3, padding=1, groups=num_L)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out += residual
        out = self.relu(out)

        return out