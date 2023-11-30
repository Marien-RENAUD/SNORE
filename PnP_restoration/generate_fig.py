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

# path_result = "/beegfs/mrenaud/Result_Average_PnP/deblurring/set4c/"

# name_list = ["Average_kernel_0","kernel_0"]
# im_name_list = ["0", "1", "2", "3"]

# im_list = []

# name_fig_list = ["Observation", "PnP Prox", "Average PnP"]

# n = 4
# m = len(im_name_list)

# #size of the black rectangle
# height = 22
# width = 140

# fig = plt.figure(figsize = (n*5, m*5))
# for i, im_name in enumerate(im_name_list):
#     dic_PnP = np.load(path_result + "PnP_Prox_k_0" + "/dict_"+im_name+"_results.npy", allow_pickle=True).item()
#     dic_AveragePnP = np.load(path_result + "Average_PnP_k_0" + "/dict_"+im_name+"_results.npy", allow_pickle=True).item()

#     gt = dic_PnP["GT"][:256,:256]
#     deblur_PnP = (dic_PnP["Deblur"][:256,:256], dic_PnP["PSNR_output"], dic_PnP["SSIM_output"], dic_PnP["LPIPS_output"])
#     blur = (dic_PnP["Blur"][:256,:256], dic_PnP["PSNR_blur"], dic_PnP["SSIM_blur"], dic_PnP["LPIPS_blur"])
#     k = dic_PnP["kernel"]
#     deblur_APnP = (dic_AveragePnP["Deblur"][:256,:256], dic_AveragePnP["PSNR_output"], dic_AveragePnP["SSIM_output"], dic_AveragePnP["LPIPS_output"])
    
#     ax = fig.add_subplot(m,n,1+4*i)
#     ax.imshow(gt)
#     rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
#     ax.add_patch(plt.Rectangle(**rect_params))
#     text_params = {'xy': (5, 5), 'text': "PSNR/SSIM/LPIPS", 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
#     ax.annotate(**text_params)
#     ax.axis('off')
#     ax.set_title("Ground Truth", fontsize=21)

#     for j, im in enumerate([blur, deblur_PnP, deblur_APnP]):
#         ax = fig.add_subplot(m,n,2+j+4*i)
#         if j ==0:
#             c = 50
#             k_resize = cv2.resize(k, dsize =(c,c), interpolation=cv2.INTER_CUBIC)
#             im[0][-k_resize.shape[0]:,:k_resize.shape[1]] = k_resize[:,:,None]*np.ones(3)[None,None,:] / np.max(k_resize)
#         ax.imshow(im[0])
#         rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
#         ax.add_patch(plt.Rectangle(**rect_params))
#         text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3]), 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
#         ax.annotate(**text_params)
#         ax.axis('off')
#         ax.set_title(name_fig_list[j], fontsize=21)

# fig.savefig(path_result+'/All_results.png')
# plt.show()


path_result = "/beegfs/mrenaud/Result_Average_PnP/deblurring/set1c/PnP_GD_k_0"

# name_list = ["Average_kernel_0","kernel_0"]
im_name_list = ["0.01", "0.05", "0.07", "0.09", "0.1", "0.5", "0.7"]

im_list = []

name_fig_list = ["Observation", "PnP Prox", "Average PnP"]

m = 1
n = len(im_name_list) + 2

#size of the black rectangle
height = 22
width = 140

fig = plt.figure(figsize = (n*5, m*5))
for i, im_name in enumerate(im_name_list):
    dic_AveragePnP = np.load(path_result + "/lamb_"+ im_name + "/dict_0_results.npy", allow_pickle=True).item()
    
    if i==0:
        gt = dic_AveragePnP["GT"][:256,:256]
        blur = (dic_AveragePnP["Blur"][:256,:256], dic_AveragePnP["PSNR_blur"], dic_AveragePnP["SSIM_blur"], dic_AveragePnP["LPIPS_blur"])
        k = dic_AveragePnP["kernel"]

        ax = fig.add_subplot(m,n,1)
        ax.imshow(gt)
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "PSNR/SSIM/LPIPS", 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
        ax.set_title("Ground Truth", fontsize=21)

        im = blur
        ax = fig.add_subplot(m,n,2)
        c = 50
        k_resize = cv2.resize(k, dsize =(c,c), interpolation=cv2.INTER_CUBIC)
        im[0][-k_resize.shape[0]:,:k_resize.shape[1]] = k_resize[:,:,None]*np.ones(3)[None,None,:] / np.max(k_resize)
        ax.imshow(im[0])
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3]), 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
        ax.set_title("Observation", fontsize=21)

    deblur_APnP = (dic_AveragePnP["Deblur"][:256,:256], dic_AveragePnP["PSNR_output"], dic_AveragePnP["SSIM_output"], dic_AveragePnP["LPIPS_output"])
    im = deblur_APnP
    ax = fig.add_subplot(m,n,i+3)
    ax.imshow(im[0])
    rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
    ax.add_patch(plt.Rectangle(**rect_params))
    text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3]), 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
    ax.annotate(**text_params)
    ax.axis('off')
    ax.set_title("$\lambda = $"+im_name, fontsize=21)

fig.savefig(path_result+'/PnP_Prox_lamb_influence.png')
plt.show()

