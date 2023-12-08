import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import utils_sr
import torch
from argparse import ArgumentParser
from utils.utils_restoration import rgb2y, psnr, array2tensor, tensor2array
from skimage.metrics import structural_similarity as ssim
from lpips import LPIPS
import sys
from matplotlib.ticker import MaxNLocator
from utils.utils_restoration import imsave, single2uint, rescale
from scipy import ndimage
from tqdm import tqdm
from time import time
from brisque import BRISQUE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_cuda_denoiser(device):
    '''
    Initialize the denoiser model with the given pretrained ckpt
    '''
    sys.path.append('../GS_denoising/')
    from lightning_GSDRUNet import GradMatch
    parser2 = ArgumentParser(prog='utils_restoration.py')
    parser2 = GradMatch.add_model_specific_args(parser2)
    parser2 = GradMatch.add_optim_specific_args(parser2)
    hparams = parser2.parse_known_args()[0]
    hparams.act_mode = hparams.act_mode_denoiser
    denoiser_model = GradMatch(hparams)
    checkpoint = torch.load(hparams.pretrained_checkpoint, map_location=device)
    denoiser_model.load_state_dict(checkpoint['state_dict'],strict=False)
    denoiser_model.eval()
    for i, v in denoiser_model.named_parameters():
        v.requires_grad = False
    denoiser_model = denoiser_model.to(device)
    return denoiser_model

denoiser_model = initialize_cuda_denoiser(device)

def denoise(x, sigma, denoiser_model, weight=1.):
    torch.set_grad_enabled(True)
    Dg, N, g = denoiser_model.calculate_grad(x, sigma)
    torch.set_grad_enabled(False)
    Dg = Dg.detach()
    N = N.detach()
    if self.hparams.rescale_for_denoising:
        N = N * (maxtmp - mintmp) + mintmp
    Dx = x - weight * Dg
    return Dx, g, Dg

