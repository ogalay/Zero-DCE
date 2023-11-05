import os
import torch
import numpy as np
from dataloader import lowlight_loader
from lowlight_test import lowlight
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS
from torch.utils.data import DataLoader
import model
import time

def get_metrics(enh_img: torch.Tensor, gt_img: torch.Tensor, device: str='cpu'):
    '''
        Функция для измерения метрик. Расчитывает среднее значение по сэмплу
        Arguments:
            :enh_img: изображения после обработки
            :gt_img: ground-truth изображения
            :device: cpu/cuda
    '''
    batch_size = enh_img.shape[0]
    enh_img = enh_img.to(device)
    gt_img = gt_img.to(device)
    # PSNR
    psnr = PSNR(data_range=batch_size).to(device)
    psnr_score = psnr(enh_img, gt_img).item()
    # SSIM
    ms_ssim = SSIM(data_range=batch_size).to(device)
    ssim_score = ms_ssim(enh_img, gt_img).item()
    # LILPS
    lpips = LPIPS('alex').to(device)
    lpips_score = lpips(enh_img, gt_img).item()
    return psnr_score, ssim_score, lpips_score


def evaluate(batch_size=8, device='cpu'):
    '''Оценка метрик решения'''

    img_names = os.listdir('data/test_data/low')
    dataset = lowlight_loader(img_names, '../Zero-DCE_code/data/test_data')
    dataloader = DataLoader(dataset, batch_size=batch_size)

    DCE_net = model.enhance_net_nopool()
    DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth', map_location=torch.device('cpu')))
    
    psnr_list = np.array([])
    ssim_list = np.array([])
    lpips_list = np.array([])
    times = np.array([])
    i = 0
    for lowlight_img, gt_img in dataloader:

        # Улучшение качества изображения
        start = time.time()
        _, enhanced_image, _ = DCE_net(lowlight_img)

        end_time = (time.time() - start)
        print(end_time)
        ################################
        psnr_score, ssim_score, lpips_score = get_metrics(enhanced_image, gt_img, device=device)

        psnr_list = np.append(psnr_list, psnr_score)
        ssim_list = np.append(ssim_list, ssim_score)
        lpips_list = np.append(lpips_list, lpips_score)
        i += 1
        print(f'{i}: PSNR {psnr_score}, SSIM: {ssim_score} LPIPS: {lpips_score}')

    print(f'PSNR: {psnr_list.mean()}\nSSIM: {ssim_list.mean()}\nLPIPS: {lpips_list.mean()}')


evaluate(2, 'cpu')



