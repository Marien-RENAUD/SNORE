import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
import os
import argparse
import cv2
import imageio

path_result = "/beegfs/mrenaud/GSPnP_2/PnP_restoration/deblurring/set3c/"

name_list = ["Average_kernel_0","kernel_0"]
im_name_list = ["0", "1", "2"]

im_list = []

fig = plt.figure(figsize = (4*5, 3*5))
for i, im_name in enumerate(im_name_list):
    dic_PnP = np.load(path_result + "kernel_0" + "/dict_"+im_name+"_results.npy", allow_pickle=True).item()
    dic_AveragePnP = np.load(path_result + "Average_kernel_0" + "/dict_"+im_name+"_results.npy", allow_pickle=True).item()

    gt = dic_PnP["GT"]
    deblur_PnP = (dic_PnP["Deblur"], dic_PnP["PSNR_output"], dic_PnP["SSIM_output"])
    blur = (dic_PnP["Blur"], dic_PnP["PSNR_blur"], dic_PnP["SSIM_blur"])
    k = dic_PnP["kernel"]
    deblur_APnP = (dic_AveragePnP["Deblur"], dic_AveragePnP["PSNR_output"], dic_AveragePnP["SSIM_output"])
    
    ax = fig.add_subplot(3,4,1+4*i)
    ax.imshow(gt)
    rect_params = {'xy': (0, 0), 'width': 100, 'height': 30, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
    ax.add_patch(plt.Rectangle(**rect_params))
    text_params = {'xy': (5, 5), 'text': "PSNR/SSIM", 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
    ax.annotate(**text_params)
    ax.axis('off')
    ax.set_title("Ground Truth", fontsize=21)

    for j, im in enumerate([blur, deblur_PnP, deblur_APnP]):
        ax = fig.add_subplot(3,4,2+j+4*i)
        im[0][-k.shape[0]:,:k.shape[1]] = k[:,:,None]*np.ones(3)[None,None,:] / np.max(k)
        ax.imshow(im[0])
        rect_params = {'xy': (0, 0), 'width': 100, 'height': 30, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}".format(im[1], im[2]), 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
        ax.set_title("Restoration", fontsize=21)

fig.savefig(path_result+'/All_results.png')
plt.show()
