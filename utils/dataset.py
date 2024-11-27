import torch
import torchvision
from PIL import Image
from torchvision import transforms
from utils.data_processing import *
import os
from torch.nn import functional as F

def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10,
                    upper_percentile=90):
    """Determine if an image is low contrast."""
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold

class FuseDataset:
    def __init__(self, vi_path: str, ir_path: str, crop_size: int) -> None:
        self.image_path = vi_path
        self.label_path = ir_path
        self.crop_size = crop_size

        self.vi = [os.path.join(vi_path, imgname) for imgname in sorted(os.listdir(vi_path))]
        self.ir = [os.path.join(ir_path, imgname) for imgname in sorted(os.listdir(ir_path))]

        self.totensor = transforms.ToTensor()
        # self.flip1 = transforms.RandomHorizontalFlip()
        # self.flip2 = transforms.RandomVerticalFlip()
        self.croper = transforms.RandomCrop(crop_size)

    def __len__(self):
        return len(self.vi)

    def __getitem__(self, index: int):
        vi = Image.open(self.vi[index]).convert('L')
        ir = Image.open(self.ir[index]).convert('L')
        factor = 1
        while vi.size[0] < self.crop_size or vi.size[1] < self.crop_size:
            vi = Image.open(self.vi[(index + factor) % len(self)]).convert('L')
            ir = Image.open(self.ir[(index + factor) % len(self)]).convert('L')
            factor += 1

        vi = self.totensor(vi)
        ir = self.totensor(ir)

        img = torch.cat([vi, ir], dim=0)
        img = self.croper(img)

        # img = self.flip1(self.flip2(img))

        vi, ir = img[:1, :, :], img[1:, :, :]

        return vi, ir
    

class TestIVDataset:
    def __init__(self, vi_path, ir_path) -> None:
        self.names = sorted(os.listdir(vi_path))
        self.vi = [os.path.join(vi_path, imgname) for imgname in sorted(os.listdir(vi_path))]
        self.ir = [os.path.join(ir_path, imgname) for imgname in sorted(os.listdir(ir_path))]

        self.totensor = transforms.ToTensor()
    
    def __len__(self):
        return len(self.vi)
    
    def compute_pad(self, x):
        rest = 16 - x % 16 if x % 16 else 0
        left = rest // 2
        return left, rest - left, left, left + x

    def __getitem__(self, index):
        vi = self.totensor(Image.open(self.vi[index]).convert('L'))
        ir = self.totensor(Image.open(self.ir[index]).convert('L'))

        left_1_pad, right_1_pad, left_1, right_1 = self.compute_pad(vi.shape[1])
        left_2_pad, right_2_pad, left_2, right_2 = self.compute_pad(vi.shape[2])

        # print(vi.shape, (left_2_pad, right_2_pad, left_1_pad, right_1_pad))
        # exit()
        vi, ir = F.pad(vi, (left_2_pad, right_2_pad, left_1_pad, right_1_pad), "constant", 0.) ,\
            F.pad(ir, (left_2_pad, right_2_pad, left_1_pad, right_1_pad), "constant", 0.)

        return vi, ir, left_1, right_1, left_2, right_2, self.names[index]
