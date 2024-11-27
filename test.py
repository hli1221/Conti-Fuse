from utils.dataset import TestIVDataset
import matplotlib.pyplot as plt
from Conti_Fuse import MyFuse
import torch
from utils.loss import loss_fun, DSimatrixLoss
import yaml
import argparse
from torch.nn.parallel import DataParallel
from torchvision import utils
import os
import tqdm
from torchvision.transforms import Grayscale
import kornia
from utils import loss

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')

device ='cuda'
loss.device = device

def main(params, trained_model, save_path, vi_path, ir_path):
    decon_loss_fun = DSimatrixLoss(window_size=11, samples= params['model']['num_K'] * 2 + 1 , sample_mode='fix_random', num_L=params['model']['num_K'] + 2, decay_rule='gaussian')
    state_dict = torch.load(trained_model, map_location='cpu')
    dataset = TestIVDataset(vi_path=vi_path,\
                             ir_path=ir_path)
    
    model = MyFuse(in_channel=1, hidden_channel=params['model']['hidden_channel'], \
                    num_L=params['model']['num_K'], num_layers=params['model']['layers'], \
                        decon_loss_fun=decon_loss_fun, use_retent=True, head=params['model']['head']).to(device)
    model.load_state_dict(state_dict['model'])
    model.to(device=device)
    model = model.eval()

    gray = Grayscale()
    logging.info('all components are ready, start to testing...')
    for i in tqdm.tqdm(range(len(dataset))):
        with torch.no_grad():
            vi, ir, left_1, right_1, left_2, right_2, name = dataset[i]

            vi, ir = vi.unsqueeze(0), ir.unsqueeze(0)
            maxx = torch.ones(size=(vi.shape[0],), dtype=torch.float32)
            minn = kornia.metrics.ssim(vi, ir, window_size=11).mean(dim=(-1, -2, -3))

            vi, ir = vi.to(device), ir.to(device)
            
            pred_img, _ = model(vi, ir, maxx, minn)
            res = gray(pred_img[0][:, left_1:right_1, left_2: right_2])
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            utils.save_image(res, \
                            os.path.join(save_path, name))
            # input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/test.yaml')
    parser.add_argument('--vi', type=str)
    parser.add_argument('--ir', type=str)
    with open(parser.parse_args().config, mode='r', encoding='utf-8') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
        main(
                params, 
                './model.pt', 
                './', 
                vi_path=r'D:\PythonProject\PyJudge\results\IVIF\MSRS\dataset\ir',
                ir_path=r'D:\PythonProject\PyJudge\results\IVIF\MSRS\dataset\vi'
        )

