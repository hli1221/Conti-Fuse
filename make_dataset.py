from torchvision import transforms
import os
from PIL import Image
import tqdm
import torch
import numpy as np
from shutil import copyfile


def is_low_contrast(image, fraction_threshold=0.9, lower_percentile=10,
                    upper_percentile=90):
    """Determine if an image is low contrast."""
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold

def crop_image(imag_path: str, label_path: str, \
               save_ir_path: str, save_vi_path: str, expand_ratio: int, crop_size: int):
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    crop_fun = transforms.RandomCrop(size=crop_size)
    
    images = [os.path.join(imag_path, imgname) for imgname in sorted(os.listdir(imag_path))]
    labels = [os.path.join(label_path, imgname) for imgname in sorted(os.listdir(label_path))]
    idx = len(os.listdir(save_ir_path))

    for image, label in tqdm.tqdm(list(zip(images, labels))):
        image = to_tensor(Image.open(image).convert('RGB'))
        label = to_tensor(Image.open(label).convert('L'))
        if image.shape[1] < crop_size or image.shape[2] < crop_size:
            continue
        for _ in range(expand_ratio):
            res = crop_fun(torch.cat([image, label], dim=0))
            cp_img, cp_label = res[:3, :, :], res[3:, :, :]

            if not (is_low_contrast(cp_img) or is_low_contrast(cp_label)):
                idx += 1
                image_name = '{}.png'.format(idx)
                to_pil(cp_img).save(os.path.join(save_vi_path, image_name))
                to_pil(cp_label).save(os.path.join(save_ir_path, image_name))


def split_dataset(source_ir, source_vi, save_train, save_test, ratio):
    ir, vi = np.array([os.path.join(source_ir, imgname) for imgname in sorted(os.listdir(source_ir))]), \
                np.array([os.path.join(source_vi, imgname) for imgname in sorted(os.listdir(source_vi))])
    
    indexes = list(range(len(ir)))
    num_train = int(len(ir) * ratio)
    np.random.shuffle(indexes)

    train_ir, train_vi, test_ir, test_vi = ir[indexes[:num_train]], vi[indexes[:num_train]],\
                                ir[indexes[num_train:]], vi[indexes[num_train:]]
    
    def make_subfolders(path):
        p1, p2 = os.path.join(path, 'ir'), os.path.join(path, 'vi')
        if not os.path.exists(p1):
            os.makedirs(p1)
        if not os.path.exists(p2):
            os.makedirs(p2)

    make_subfolders(save_train)
    make_subfolders(save_test)

    def copy_files(source, target):
        for file in source:
            file_name = file.split(os.sep)[-1]
            copyfile(file, os.path.join(target, file_name))
    
    copy_files(train_ir, os.path.join(save_train, 'ir'))
    copy_files(train_vi, os.path.join(save_train, 'vi'))
    copy_files(test_ir, os.path.join(save_test, 'ir'))
    copy_files(test_vi, os.path.join(save_test, 'vi'))

# split_dataset(source_ir='./dataset/RoadScene/ir', source_vi='./dataset/RoadScene/vi', \
#               save_train='./dataset/RoadScene/train', save_test='./dataset/RoadScene/test', ratio=0.8)

crop_image(imag_path='./dataset/MSRS/train/vi', label_path='./dataset/MSRS/train/ir', \
           save_ir_path='./dataset/Train/ir', save_vi_path='./dataset/Train/vi', crop_size=192, expand_ratio=10)