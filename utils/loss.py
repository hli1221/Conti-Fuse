import torch
from math import exp
import kornia
from torch import Tensor
from einops import rearrange
from torch import nn
from torch.utils import checkpoint
from typing import *
import random
import logging
import torch.nn.functional as F
import os

device = 0

logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')


def cc(img1, img2):
    """
    img1: (C, H, W)
    """
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = img1.shape
    img1 = img1.reshape(N, C, -1)
    img2 = img2.reshape(N, C, -1)
    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)
    cc = torch.sum(img1 * img2, dim=-1) / (
        eps
        + torch.sqrt(torch.sum(img1**2, dim=-1))
        * torch.sqrt(torch.sum(img2**2, dim=-1))
    )
    cc = torch.clamp(cc, -1.0, 1.0)
    # print(cc.shape)
    return cc.mean(-1)

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda(device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda(device)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)

mse = torch.nn.MSELoss()
Loss_ssim = kornia.losses.ssim.SSIMLoss(11, reduction='mean')
sigmoid = torch.nn.Sigmoid()
sobel = Sobelxy()
sobel = sobel.requires_grad_(False)

def loss_fun(pred_img, ir, vi):
    x_in_max = torch.max(ir, vi)
    loss_in = F.l1_loss(x_in_max, pred_img)
    y_grad = sobel(vi)
    ir_grad = sobel(ir)
    generate_img_grad = sobel(pred_img)
    x_grad_joint = torch.max(y_grad, ir_grad)
    loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
    return loss_in, loss_grad


def ssim_simlarity(img1: Tensor, img2: Tensor, window_size: int=7):
    """
    img1: (B C H W)
    img2: (B C H W)
    """
    res = cc(img1, img2)
    return res

def compute_simatrix(transit_unit: Tensor, window_size: int=11):
    """
    transit_unit: (B L C H W)
    """
    B, L, C, H, W = transit_unit.shape
    transit_unit = torch.transpose(transit_unit, 0, 1) # (L B C H W)
    all = rearrange(transit_unit, "L B C H W -> (L B) C H W")
    res = torch.zeros(size=(L, L, B), device='cuda:{}'.format(device))

    def handdle(img1):
        img1 = img1.unsqueeze(0).expand(size=(L, B, C, H, W))
        img1 = rearrange(now.unsqueeze(0).expand(size=(L, B, C, H, W))\
                                                               , "L B C H W -> (L B) C H W")
        return ssim_simlarity(img1, all, window_size)
    
    for i, now in enumerate(transit_unit):
        temp = checkpoint.checkpoint(handdle, now)
        res[i] += rearrange(temp, "(L B) -> L B", L=L)

    return res.permute(2, 0, 1)


def get_matrix_gt(size: int, max: int, min: int, decay_rule: str='linear'):
    max = max.unsqueeze(1).unsqueeze(1)
    min = min.unsqueeze(1).unsqueeze(1)
    t1, t2 = torch.arange(size).unsqueeze(0), torch.arange(size).unsqueeze(1)
    pos_matrix = (t1 - t2).abs().unsqueeze(0)
    if decay_rule == 'linear':
        weight = (max - min) / (size - 1)
        return (max - pos_matrix * weight)
    elif decay_rule == 'gaussian':
        sigma_square =  - pos_matrix.max() ** 2 / (2 * torch.log(torch.tensor(min)))
        return torch.exp(-(pos_matrix) ** 2 / (2 * sigma_square))


def compute_simatrix(transit_unit: Tensor, window_size: int=11):
    """
    transit_unit: (B L C H W)
    """
    B, L, C, H, W = transit_unit.shape
    transit_unit = torch.transpose(transit_unit, 0, 1) # (L B C H W)
    all = rearrange(transit_unit, "L B C H W -> (L B) C H W")
    res = torch.zeros(size=(L, L, B), device='cuda:{}'.format(device))

    def handdle(img1):
        img1 = img1.unsqueeze(0).expand(size=(L, B, C, H, W))
        img1 = rearrange(now.unsqueeze(0).expand(size=(L, B, C, H, W))\
                                                               , "L B C H W -> (L B) C H W")
        return ssim_simlarity(img1, all, window_size)
    
    for i, now in enumerate(transit_unit):
        temp = checkpoint.checkpoint(handdle, now)
        res[i] += rearrange(temp, "(L B) -> L B", L=L)

    return res.permute(2, 0, 1)


class DSimatrixLoss(nn.modules.loss._Loss):
    def __init__(self, window_size: int, samples: Union[int, float], sample_mode: str, \
                 num_L: int, decay_rule: str='linear') -> None:
        """
        sample_mode: random | fix_random | diag
        """
        super().__init__()
        if isinstance(samples, float):
            self.samples = int(num_L * (num_L - 1) // 2 - 1 * samples)
        else:
            self.samples = samples

        self.window_size = window_size
        self.decay_rule = decay_rule
        self.sample_mode = sample_mode
        self.num_L = num_L
        self.random_set = [(i // num_L, i % num_L) if i % num_L > i // num_L else (i % num_L, i // num_L) \
                               for i in range(num_L ** 2) if i // num_L != i % num_L]
        self.random_set = sorted(list(set(self.random_set)), key=lambda x: abs(x[0] - x[1]))[:-1]

    def sample_examples(self):
        if self.sample_mode == 'fix_random':
            fix = self.random_set[:self.num_L - 1]
            rands = random.sample(self.random_set[self.num_L - 1:], self.samples - self.num_L + 1)
            samples = fix + rands
            samples_x = [i for i, j in samples]
            samples_y = [j for i, j in samples]
            # logging.info(str(samples))
            return torch.LongTensor(samples_x).cuda(device), torch.LongTensor(samples_y).cuda(device)

    def forward(self, transit_unit: Tensor, max, min):
        B, L, C, H, W = transit_unit.shape
        samples_x, samples_y = self.sample_examples()
        
        x, y = transit_unit[:, samples_x, :, :, :], transit_unit[:, samples_y, :, :, :]
        x, y = x.reshape(-1, C, H, W), y.reshape(-1, C, H, W)

        ssim = ssim_simlarity(x, y, window_size=self.window_size)

        gt = get_matrix_gt(size=self.num_L, max=max, min=min, decay_rule=self.decay_rule).cuda(device)
        gt = gt[0][samples_x, samples_y].expand(size=(B, len(samples_y))).reshape(-1)

        return ((ssim - gt) ** 2).reshape(B, -1).mean(dim=-1)
























# from matplotlib import pyploto as plt

# linear = get_matrix_gt(64)
# gaussian = get_matrix_gt(64, decay_rule='gaussian')

# fg1 = plt.subplot(1, 2, 1)
# im1 = fg1.imshow(linear[0], cmap=plt.cm.hot_r)
# plt.colorbar(im1)
# plt.title('linear')

# fg2 = plt.subplot(1, 2, 2)
# im2 = fg2.imshow(gaussian[0], cmap=plt.cm.hot_r)
# plt.colorbar(im2)
# plt.title('gaussian')
# plt.show()
