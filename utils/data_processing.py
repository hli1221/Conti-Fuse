import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import os
import tqdm
from PIL import Image
from typing import *
from torch.distributions import bernoulli, Normal
from torch.nn import functional
import torch
from torchvision import transforms

plt.subplots_adjust(wspace = 0, hspace = 1)#调整子图间距

def sample_mask(ratio: float, image_size: int, tp_size: Tuple[int, int, int]):
    dist1 = bernoulli.Bernoulli(probs=torch.ones(tp_size[0] ** 2) * ratio)
    dist2 = bernoulli.Bernoulli(probs=torch.ones(tp_size[1] ** 2) * ratio)
    dist3 = bernoulli.Bernoulli(probs=torch.ones(tp_size[2] ** 2) * ratio)
    
    mask = functional.interpolate(dist1.sample().reshape(1, 1, tp_size[0], tp_size[0])\
                                  , size=image_size, mode='nearest').squeeze().type(torch.int32)
    mask |= functional.interpolate(dist2.sample().reshape(1, 1, tp_size[1], tp_size[1])\
                                  , size=image_size, mode='nearest').squeeze().type(torch.int32)
    mask |= functional.interpolate(dist3.sample().reshape(1, 1, tp_size[2], tp_size[2])\
                                  , size=image_size, mode='nearest').squeeze().type(torch.int32)
    return mask


def sample_normal(sigma=0.06, mu=0.1665, bound_l=0, bound_r=0.3333):
    sampler = Normal(loc=mu, scale=sigma)
    res = sampler.sample()
    while res > bound_r or res < bound_l:
        res = sampler.sample()
    return res


def make_mask(mask_ratio_range: Tuple[float, float], \
              image_size: int, tp_size: Tuple[int, int, int]):
    
    assert 0 <= mask_ratio_range[0] < mask_ratio_range[1] <= 1, 'mask_ratio is invalid, \
        expect the mask ratio is between 0-1 and ratio1 < ratio2, but got {} and {}'.format(mask_ratio_range[0], mask_ratio_range[1])

    mask_all = random.random() * (mask_ratio_range[1] - mask_ratio_range[0]) + mask_ratio_range[0]
    mask_part = sample_normal()

    mask_1 = sample_mask(mask_all, image_size=image_size, tp_size=tp_size)
    mask_2 = sample_mask(mask_part, image_size=image_size, tp_size=tp_size)

    mask_2 = mask_1 & mask_2
    mask_1 -= mask_2

    return mask_1, mask_2


def make_mask_fill(source_image, tp_size, image_size, ratio):
    """
    gaussion, blur, fill

    1. select ratio
    2. select region
    3. select own fill ratio
    """
    blurer = transforms.GaussianBlur(kernel_size=random.randint(5, 20) * 2 + 1, sigma=[10, 30])

    own_ratio_1 = sample_normal(0.1, mu=0.5, bound_l=0, bound_r=1)

    mask_1 = sample_mask(ratio, image_size, tp_size)

    gaussion_mask = source_image * own_ratio_1 + torch.randn_like(source_image).abs() * (1 - own_ratio_1)
    blur_mask = blurer(source_image)

    res = gaussion_mask * (1 - mask_1) + mask_1 * blur_mask
    return res


# while True:
#     cnt = 0
#     for i in range(5):
#         mask1, mask2 = make_mask(mask_ratio_range=[0.1, 0.16], image_size=224, tp_size=[13, 26, 26])
#         mask1 = mask1.squeeze(0)
#         mask2 = mask2.squeeze(0)
#         all = mask2 * mask1
        
#         plt.subplot(5, 3, cnt + 1)
#         cnt += 1
#         plt.imshow(1 - mask1)

#         plt.subplot(5, 3, cnt + 1)
#         cnt += 1
#         plt.imshow(1 - mask2)

#         plt.subplot(5, 3, cnt + 1)
#         cnt += 1
#         plt.imshow((1 - mask1) | (1 - mask2))
#     plt.show()


def crop_image(imag_path: str, label_path: str, threshould: float, \
               save_img_path: str, save_label_path: str, expand_ratio: int, crop_size: int):

        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        crop_fun = transforms.RandomCrop(size=crop_size)
        
        images = [os.path.join(imag_path, imgname) for imgname in sorted(os.listdir(imag_path))]
        labels = [os.path.join(label_path, imgname) for imgname in sorted(os.listdir(label_path))]
        idx = 0
        for image, label in tqdm.tqdm(list(zip(images, labels))):
            image = to_tensor(Image.open(image).convert('RGB'))
            label = to_tensor(Image.open(label).convert('L'))
            if image.shape[1] < crop_size or image.shape[2] < crop_size:
                continue
            for _ in range(expand_ratio):
                res = crop_fun(torch.cat([image, label], dim=0))
                cp_img, cp_label = res[:3, :, :], res[3:, :, :]
                ratio_x = cp_label.sum() / crop_size ** 2
                ratio_y = 1 - ratio_x
                if ratio_x > threshould and ratio_y > threshould:
                    idx += 1
                    image_name = '{}.png'.format(idx)
                    to_pil(cp_img).save(os.path.join(save_img_path, image_name))
                    to_pil(cp_label).save(os.path.join(save_label_path, image_name))

if __name__ == '__main__':
    pass
    # crop_image(imag_path=r'./datasets/COCO/val2017', label_path=r'./datasets/COCO/val_labels',
    #            threshould=0.2, save_img_path=r'./datasets/COCO/cropped_images', save_label_path=r'./datasets/COCO/cropped_labels'\
    #             , expand_ratio=5, crop_size=192)