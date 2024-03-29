import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import utils_sr
import torch
from argparse import ArgumentParser
from utils.utils_restoration import rgb2y, psnr, array2tensor, tensor2array, imread_uint
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import estimate_sigma
from lpips import LPIPS
import sys
from matplotlib.ticker import MaxNLocator
from utils.utils_restoration import imsave, single2uint, rescale
from scipy import ndimage
from tqdm import tqdm
from time import time
from brisque import BRISQUE

path_ckpt = '../GS_denoising/ckpts/GSDRUNet_grayscale.ckpt' #test/epoch=762-step=16786.ckpt'
device = 'cuda:1'

def initialize_cuda_denoiser():
    '''
    Initialize the denoiser model with the given pretrained ckpt
    '''
    sys.path.append('../GS_denoising/')
    from lightning_GSDRUNet import GradMatch
    parser2 = ArgumentParser(prog='utils_restoration.py')
    parser2 = GradMatch.add_model_specific_args(parser2)
    parser2 = GradMatch.add_optim_specific_args(parser2)
    hparams = parser2.parse_known_args()[0]
    hparams.act_mode = 'E'
    hparams.grayscale = True
    denoiser_model = GradMatch(hparams)
    checkpoint = torch.load(path_ckpt, map_location=device)
    denoiser_model.load_state_dict(checkpoint['state_dict'],strict=False)
    denoiser_model.eval()
    for i, v in denoiser_model.named_parameters():
        v.requires_grad = False
    denoiser_model = denoiser_model.to(device)
    return denoiser_model

def denoise(x, sigma, weight=1.):
    if rescale_for_denoising:
        mintmp = x.min()
        maxtmp = x.max()
        x = (x - mintmp) / (maxtmp - mintmp)
    elif clip_for_denoising:
            x = torch.clamp(x,0,1)
    torch.set_grad_enabled(True)
    Dg, N, g = denoiser_model.calculate_grad(x, sigma)
    torch.set_grad_enabled(False)
    Dg = Dg.detach()
    N = N.detach()
    if rescale_for_denoising:
        N = N * (maxtmp - mintmp) + mintmp
    Dx = x - weight * Dg
    return Dx, g, Dg

denoiser_model = initialize_cuda_denoiser()
rescale_for_denoising = False
clip_for_denoising = True

path_result = "../../Result_SAR/"

for i in tqdm(range(10)):
    for sigma in [5., 10., 20., 40.]:
        path_result_im = path_result + "sigma="+str(sigma)+"/"
        if not os.path.exists(path_result_im):
            os.mkdir(path_result_im)
        input_im_uint = imread_uint("../datasets/SAR_test/SAR/"+str(i)+".png", n_channels=1)
        input_im = np.float32(input_im_uint / 255.)
        input_im_t = array2tensor(input_im).to(device)

        # sigma = 10.
        noise = np.random.normal(0, sigma / 255., input_im.shape)
        im_noised = input_im + noise
        print(np.max(im_noised))
        im_noised =  np.float32(im_noised)
        im_noised_t = array2tensor(im_noised).to(device)

        im_denoised_t, _, _ = denoise(im_noised_t, sigma / 255.)
        im_denoised = tensor2array(im_denoised_t)

        plt.imsave(path_result_im + str(i) + '_im.png', input_im[:,:,0])
        plt.imsave(path_result_im + str(i) + '_im_noised.png', im_noised[:,:,0])
        plt.imsave(path_result_im + str(i) + '_im_denoised.png', im_denoised[:,:,0])

        loss_lpips = LPIPS(net='alex', version='0.1').to(device)

        print("PSNR :", psnr(input_im, im_denoised))
        print("SSIM :", ssim(input_im, im_denoised, data_range = 1, channel_axis = 2))
        print("LPIPS :", loss_lpips.forward(input_im_t, im_denoised_t).item())

        dict = {
                'im' : input_im[:,:,0],
                'im_noised' : im_noised[:,:,0],
                'im_denoised' : im_denoised[:,:,0],
                'LPIPS' : loss_lpips.forward(input_im_t, im_denoised_t).item(),
                'PSNR' : psnr(input_im, im_denoised),
                'SSIM' : ssim(input_im, im_denoised, data_range = 1, channel_axis = 2),
                'sigma' : sigma,
            }
        np.save(os.path.join(path_result_im, 'dict_' + str(i) + '_results'), dict)