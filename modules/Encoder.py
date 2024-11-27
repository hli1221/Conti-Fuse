from torch import nn 
from torch.utils import checkpoint

class EncoderLayer(nn.Module):
    def __init__(self, in_channel: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channel * 2, out_channels=in_channel * 2, kernel_size=3, padding=1)

        self.none_linear = nn.GELU()
        self.in_channel = in_channel
        self.pooling = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        x = self.none_linear(self.conv1(x))
        x = self.none_linear(self.conv2(x))
        return self.pooling(x)

class Encoder(nn.Module):
    def __init__(self, num_layers: int=5, in_channel:int = 3, use_retent: bool=False) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            EncoderLayer(in_channel * 2 ** i) for i in range(num_layers)
        )
        self.use_retent = use_retent
    
    def forward(self, x):
        skips = []
        for layer in self.layers:
            if self.use_retent:
                x = checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = checkpoint.checkpoint(layer, x)
            skips.append(x)
        return skips[-1], skips[:-1]