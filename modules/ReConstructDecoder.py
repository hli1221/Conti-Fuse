from torch import nn
from einops import rearrange
import torch
import matplotlib.pyplot as plt
from torch.utils import checkpoint
import numbers
import matplotlib.pyplot as plt
from modules.StateTransformer import StateTransformer
from modules.ResNet import *

class CDM(nn.Module):
    def __init__(self, num_L: int, in_channel: int, head: int) -> None:
        super().__init__()
        self.num_L = num_L
        self.in_channel = in_channel

        self.shallow_channel = in_channel
        
        self.P = BasicBlock(in_chan=self.shallow_channel * num_L, out_chan=self.shallow_channel * num_L, num_L=num_L)
        self.statetrans = StateTransformer(channel=self.shallow_channel, num_L=num_L, head=head)

        self.linear_trans = nn.Conv2d(in_channels=in_channel * 2, out_channels=self.shallow_channel * num_L, padding=1, kernel_size=3)
    
    def forward(self, x, y, maxx, minn, decon_loss_fun, layer):
        states = self.linear_trans(torch.cat([x, y], dim=1))

        states = self.P(states)
        states = rearrange(states, 'B (L C) H W -> B L C H W', L=self.num_L)
        states = self.statetrans(states)
        
        # decon_ = torch.cat([x.unsqueeze(1), states, y.unsqueeze(1)], dim=1)
        # decon_loss = decon_loss_fun(decon_, maxx, minn)
        decon_loss = 0

        return states, decon_loss


class ReconstructDecoderLayer(nn.Module):
    def __init__(self, in_channel: int, num_L: int, head: int) -> None:
        super().__init__()
        self.dynamic = CDM(num_L=num_L, in_channel=in_channel // 2, head=head)

        self.scale_down = nn.Conv2d(in_channels=in_channel // 2 * (num_L + 2), out_channels=in_channel, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(in_channels=in_channel * 2, out_channels=in_channel // 2, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=in_channel // 2, out_channels=in_channel // 2, padding=1, kernel_size=3)

        self.none_linear = nn.GELU()
        self.in_channel = in_channel
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x, y, last, maxx, minn, decon_loss_fun, layer):
        """
        x: (B, C, H, W)
        y: (B, C, H, W)
        last: (B, C * 2, H, W)
        """
        last = self.upsample(last)

        c_x, c_y = torch.clone(x).detach(), torch.clone(y).detach()
        res, decon_loss = self.dynamic(c_x, c_y, maxx, minn, decon_loss_fun, layer)

        res = self.scale_down(torch.cat([x, rearrange(res, "B L C H W -> B (L C) H W"), y], dim=1))
        res = self.none_linear(self.conv1(torch.cat([res, last], dim=1)))
        res = self.conv2(res)
        return res, decon_loss


class ReconstructDecoder(nn.Module):
    def __init__(self, num_layers: int, in_channel: int, num_L: int, \
                 decon_loss_fun, head, use_retent:bool=False) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [ReconstructDecoderLayer(num_L=num_L, \
                                     in_channel=in_channel // (2 ** i), head=head) for i in range(num_layers)]
        )
        self.decon_loss_fun = decon_loss_fun
        self.use_retent = use_retent
    
    def forward(self, x, skips, maxx, minn):
        """
        last: (B, 1024, 14, 14) # need upsample and conv
        x: (B, 512, 14, 14) # need upsample
        y: (B, 512, 14, 14)
        mask (4, 1024, 28, 28) # need conv
        """
        last = x
        all_loss = 0

        for i, (skip, layer) in enumerate(zip(list(reversed(skips)), self.layers)):
            B = skip.shape[0] // 2
            x, y = skip[: B, :, :, :], skip[B:, :, :, :]
            if self.use_retent:
                last, decon_loss = checkpoint.checkpoint(layer, x, y, last, \
                                                        maxx, minn, self.decon_loss_fun, i, use_reentrant=False)
            else:
                last, decon_loss = checkpoint.checkpoint(layer, x, y, last, maxx, minn, self.decon_loss_fun, i)
            all_loss += decon_loss
        return last, all_loss
            
