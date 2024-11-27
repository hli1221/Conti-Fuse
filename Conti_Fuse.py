import torch
from torch import nn
from modules.Encoder import *
from modules import ReConstructDecoder


class MyFuse(nn.Module):
    def __init__(self, in_channel: int, hidden_channel: int,\
                  num_L: int, num_layers: int, decon_loss_fun, \
                    head, use_retent: bool=True) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels=in_channel, out_channels=hidden_channel, kernel_size=1)

        self.out_proj_recon = nn.Conv2d(in_channels=hidden_channel, out_channels=in_channel, kernel_size=1)

        self.middle_channels = hidden_channel * 2 ** num_layers
        self.Encoder = Encoder(num_layers=num_layers, in_channel=hidden_channel, use_retent=use_retent)
        self.ReConstructDecoder = ReConstructDecoder.ReconstructDecoder(num_layers=num_layers, in_channel=self.middle_channels, \
                                                      num_L=num_L, decon_loss_fun=decon_loss_fun, use_retent=use_retent, head=head)
        self.sig = nn.Sigmoid()
        
        
    def forward(self, x, y, maxx, minn):
        B = x.shape[0]
        ipt = torch.cat([x, y], dim=0)
        ipt = self.in_proj(ipt)

        x, skips = self.Encoder(ipt)
        skips = [ipt] + skips
        recon_in = x[:B, :, :, :] + x[B:, :, :, :]

        fused, all_loss = self.ReConstructDecoder(recon_in, skips, maxx, minn)
        return self.sig(self.out_proj_recon(fused)), all_loss
