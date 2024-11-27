from torch import nn 
from torch.utils import checkpoint

class MaskDecoderLayer(nn.Module):
    def __init__(self, in_channel: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channel // 2, out_channels=in_channel // 2, kernel_size=3, padding=1)
        self.none_linear = nn.GELU()
        self.in_channel = in_channel
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.none_linear(self.conv1(x))
        x = self.none_linear(self.conv2(x))
        return x

class MaskDecoder(nn.Module):
    def __init__(self, in_channel, num_layers: int=5, use_retent:bool=False) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            MaskDecoderLayer(in_channel // (2 ** i)) for i in range(num_layers)
        )
        self.use_retent = use_retent
    
    def forward(self, x):
        masks = []
        for layer in self.layers:
            if self.use_retent:
                x = checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = checkpoint.checkpoint(layer, x)
            masks.append(x)
        return x, masks