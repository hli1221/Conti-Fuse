import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F
    
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNorm, self).__init__()

        self.weight = nn.Parameter(torch.ones(size=(1, 1, normalized_shape, 1, 1)))
        self.bias = nn.Parameter(torch.zeros(size=(1, 1, normalized_shape, 1, 1)))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(2, keepdim=True)
        sigma = x.var(2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class StateAttention(nn.Module):
    def __init__(self, head, channel, num_L) -> None:
        super().__init__()
        self.head = head
        self.qkv = nn.Conv2d(in_channels=channel, out_channels=channel * 3, kernel_size=3, padding=1)
        self.num_L = num_L
        self.project = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)

    
    def forward(self, x):
        """
        x: (B L C H W)
        """
        B, L, C, H, W = x.shape
        q, k, v = [rearrange(p, "(B L) C H W -> B L (C H W) ", L=self.num_L) for p in torch.chunk(
            self.qkv(rearrange(x, "B L C H W -> (B L) C H W"))
        , dim=1, chunks=3)]

        q, k, v = [rearrange(p, "B L (h w) -> B h L w", h = self.head) for p in [q, k, v]]
        temperature = (C * H * W  / self.head) ** -0.5

        atten = torch.softmax((q @ k.transpose(-1, -2)) * temperature, dim=-1)
        out = atten @ v

        out = rearrange(out, "B h L w -> B L (h w)")

        out = self.project(
            rearrange(out, "B L (C H W) -> (B L) C H W", H=H, W=W)
        )
        return rearrange(out, "(B L) C H W -> B L C H W", L=self.num_L)


class FeedForward(nn.Module):
    def __init__(self, dim, num_L, ffn_expansion_factor=2):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2
        )

        self.num_L = num_L

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1)

    def forward(self, x):
        x = rearrange(x, "B L C H W -> (B L) C H W")
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        x = rearrange(x, "(B L) C H W -> B L C H W", L=self.num_L)
        return x

class StateTransformer(nn.Module):
    def __init__(self, head, channel, num_L):
        super(StateTransformer, self).__init__()
        self.norm1 = LayerNorm(channel)  # 层归一化
        self.attn = StateAttention(channel=channel, head=head, num_L=num_L)
        self.norm2 = LayerNorm(channel)
        self.ffn = FeedForward(channel, num_L=num_L)
        self.num_L = num_L

    def forward(self, x):
        x = x + self.norm1(self.attn(x))
        x = x + self.norm2(self.ffn(x))
        return x
