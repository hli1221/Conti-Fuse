# import torch
# import torch.nn.functional as F
# from einops import rearrange

# x = torch.randn(size=(1, 4, 5, 5))
# conv = torch.nn.Conv2d(in_channels=4, out_channels=4, groups=2, kernel_size=3, bias=False)
# a = conv(x)

# p = rearrange(x, 'B (L C) H W -> B L C H W', L=2)
# b1, b2 = p[:, 0, :, :, :], p[:, 1, :, :, :]

# b1, b2 = F.conv2d(b1, conv.weight[:2, :, :, :]), F.conv2d(b2, conv.weight[2:, :, :, :])
# b = torch.cat([b1, b2], dim=1)
# print(a == b)