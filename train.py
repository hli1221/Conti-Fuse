import torch

from utils.dataset import FuseDataset
from torch.utils.data import DataLoader
from utils.lr_scheduler import WarmupCosineSchedule
from UnifiedFuse import MyFuse
from utils.loss import loss_fun, DSimatrixLoss
import yaml
import argparse
from torch.utils.tensorboard import SummaryWriter
import logging
import os
from torch import nn
import kornia
from utils import loss


# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# torch.cuda.set_device(0)

device = 2
logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
loss.device = device

def main(params):
    sm_writter = SummaryWriter('./logs/with_SDS')
    dataset = FuseDataset(vi_path='./dataset/MSRS/train/vi', ir_path='./dataset/MSRS/train/ir', \
                          crop_size=params['model']['crop_size'])
    loader = DataLoader(dataset=dataset, batch_size=params['training']['batch_size'], shuffle=True, num_workers=12)

    decon_loss_fun = DSimatrixLoss(window_size=11, samples= 31 , sample_mode='fix_random', num_L=params['model']['num_K'] + 2, decay_rule='gaussian')

    model = MyFuse(in_channel=1, hidden_channel=params['model']['hidden_channel'], \
                    num_L=params['model']['num_K'], num_layers=params['model']['layers'], \
                        decon_loss_fun=decon_loss_fun, use_retent=True, head=params['model']['head']).to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)
    lr_scheduler = WarmupCosineSchedule(optimizer=optimizer, warmup_steps=params['training']['warmup'] * len(loader),
                                        start_lr=params['training']['start_lr'], ref_lr=params['training']['ref_lr'],\
                                            final_lr=params['training']['final_lr'], T_max=len(loader) * params['training']['epoch'])


    logging.info('all components are ready, start training...')
    for epoch in range(params['training']['epoch']):
        logging.info('start to training on {}'.format(epoch))
        all_loss = 0
        all_in = 0
        all_grad = 0
        all_decom = 0
        if epoch > 250:
            exit()
        for i, (vi, ir) in enumerate(loader):
            optimizer.zero_grad()
            model.zero_grad()
            
            maxx = torch.ones(size=(vi.shape[0],), dtype=torch.float32)
            minn = kornia.metrics.ssim(vi, ir, window_size=11).mean(dim=(-1, -2, -3))
            
            vi, ir = vi.to(device), ir.to(device)

            pred_img, decon_loss = model(vi, ir, maxx, minn)
            decon_loss = decon_loss.mean()
            loss_in, loss_grad = loss_fun(pred_img, ir, vi)
            
            loss = decon_loss + loss_in * params['model']['factor1'] + loss_grad * params['model']['factor2']

            with torch.no_grad():
                all_loss += loss.item()
                all_in += loss_in.item()
                all_grad += loss_grad.item()
                all_decom += decon_loss.item()

            loss.backward()
            nn.utils.clip_grad.clip_grad_norm_(model.parameters(),  params['model']['clip'], norm_type=2)
            optimizer.step()
            
            lr_scheduler.step()

            
        sm_writter.add_scalar('loss', all_loss, epoch)
        sm_writter.add_scalar('lr', lr_scheduler.lr, epoch)
        sm_writter.add_scalar('recon', all_in, epoch)
        sm_writter.add_scalar('grad', all_grad, epoch)
        sm_writter.add_scalar('decon_loss', all_decom, epoch)


        state_dict = {
            'model': model.eval().state_dict(),
            'scheduler': lr_scheduler,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        save_pth = './trained_models/with_SDS/'
        if not os.path.exists(save_pth):
            os.mkdir(save_pth)

        torch.save(state_dict, os.path.join(save_pth, 'epoch_{}.pt'.format(epoch)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/UF-Base.yaml')
    with open(parser.parse_args().config, mode='r', encoding='utf-8') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
        main(params)

