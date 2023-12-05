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

path_result = "/beegfs/mrenaud/Result_Average_PnP/deblurring/set4c/"

# name_list = ["Average_PnP_k_0/noise_10.0","PnP_Prox_k_0/noise_10.0"]
im_name_list = ["0", "1", "2", "3"]

im_list = []

name_fig_list = ["Observation", "PnP Prox", "Average PnP", "Average PnP Prox"]

n = len(name_fig_list) + 1
m = len(im_name_list)

#size of the black rectangle
height = 22
width = 220

fig = plt.figure(figsize = (n*5, m*5))
for i, im_name in enumerate(im_name_list):
    width = 220
    dic_PnP = np.load(path_result + "PnP_Prox_k_0/noise_10.0/dict_"+im_name+"_results.npy", allow_pickle=True).item()
    dic_APnP = np.load(path_result + "Average_PnP_k_0/noise_10.0/dict_"+im_name+"_results.npy", allow_pickle=True).item()
    dic_APnPProx = np.load(path_result + "Average_PnP_Prox_k_0/noise_10.0/dict_"+im_name+"_results.npy", allow_pickle=True).item()

    gt = dic_PnP["GT"][:256,:256]
    deblur_PnP = (dic_PnP["Deblur"][:256,:256], dic_PnP["PSNR_output"], dic_PnP["SSIM_output"], dic_PnP["LPIPS_output"], dic_PnP["BRISQUE_output"])
    blur = (dic_PnP["Blur"][:256,:256], dic_PnP["PSNR_blur"], dic_PnP["SSIM_blur"], dic_PnP["LPIPS_blur"], dic_PnP["BRISQUE_blur"])
    k = dic_PnP["kernel"]
    deblur_APnP = (dic_APnP["Deblur"][:256,:256], dic_APnP["PSNR_output"], dic_APnP["SSIM_output"], dic_APnP["LPIPS_output"], dic_APnP["BRISQUE_output"])
    deblur_APnPProx = (dic_APnPProx["Deblur"][:256,:256], dic_APnPProx["PSNR_output"], dic_APnPProx["SSIM_output"], dic_APnPProx["LPIPS_output"], dic_APnPProx["BRISQUE_output"])

    ax = fig.add_subplot(m,n,1+n*i)
    ax.imshow(gt)
    rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
    ax.add_patch(plt.Rectangle(**rect_params))
    text_params = {'xy': (5, 5), 'text': "PSNR/SSIM/LPIPS/BRISQUE", 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
    ax.annotate(**text_params)
    ax.axis('off')
    ax.set_title("Ground Truth", fontsize=21)

    width = 190

    for j, im in enumerate([blur, deblur_PnP, deblur_APnP, deblur_APnPProx]):
        ax = fig.add_subplot(m,n,2+j+n*i)
        if j ==0:
            c = 50
            k_resize = cv2.resize(k, dsize =(c,c), interpolation=cv2.INTER_CUBIC)
            im[0][-k_resize.shape[0]:,:k_resize.shape[1]] = k_resize[:,:,None]*np.ones(3)[None,None,:] / np.max(k_resize)
        ax.imshow(im[0])
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
        ax.set_title(name_fig_list[j], fontsize=21)

fig.savefig(path_result+'/All_results_PnP_Prox_Average_PnPnProx.png')
plt.show()

# path_result = "/beegfs/mrenaud/Result_Average_PnP/deblurring/set1c/Average_PnP_k_0/noise_10.0"

# # name_list = ["Average_kernel_0","kernel_0"]
# im_name_list = ["1", "2", "3", "4", "5", "10"]
# # ["0.05", "0.1", "0.3", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "5.0", "10.0", "20.0"]

# im_list = []

# name_fig_list = ["Observation", "PnP Prox", "Average PnP"]

# m = 1
# n = len(im_name_list) + 2

# #size of the black rectangle
# height = 22
# width = 220

# fig = plt.figure(figsize = (n*5, m*5))
# for i, im_name in enumerate(im_name_list):
#     dic_AveragePnP = np.load(path_result + "/num_noise_"+ im_name + "/dict_0_results.npy", allow_pickle=True).item()
    
#     if i==0:
#         gt = dic_AveragePnP["GT"][:256,:256]
#         blur = (dic_AveragePnP["Blur"][:256,:256], dic_AveragePnP["PSNR_blur"], dic_AveragePnP["SSIM_blur"], dic_AveragePnP["LPIPS_blur"], dic_AveragePnP["BRISQUE_blur"])
#         k = dic_AveragePnP["kernel"]

#         ax = fig.add_subplot(m,n,1)
#         ax.imshow(gt)
#         rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
#         ax.add_patch(plt.Rectangle(**rect_params))
#         text_params = {'xy': (5, 5), 'text': "PSNR/SSIM/LPIPS/BRISQUE", 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
#         ax.annotate(**text_params)
#         ax.axis('off')
#         ax.set_title("Ground Truth", fontsize=21)

#         im = blur
#         ax = fig.add_subplot(m,n,2)
#         c = 50
#         k_resize = cv2.resize(k, dsize =(c,c), interpolation=cv2.INTER_CUBIC)
#         im[0][-k_resize.shape[0]:,:k_resize.shape[1]] = k_resize[:,:,None]*np.ones(3)[None,None,:] / np.max(k_resize)
#         ax.imshow(im[0])
#         rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
#         ax.add_patch(plt.Rectangle(**rect_params))
#         text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
#         ax.annotate(**text_params)
#         ax.axis('off')
#         ax.set_title("Observation", fontsize=21)

#     width = 190

#     deblur_APnP = (dic_AveragePnP["Deblur"][:256,:256], dic_AveragePnP["PSNR_output"], dic_AveragePnP["SSIM_output"], dic_AveragePnP["LPIPS_output"], dic_AveragePnP["BRISQUE_output"])
#     im = deblur_APnP
#     ax = fig.add_subplot(m,n,i+3)
#     ax.imshow(im[0])
#     rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
#     ax.add_patch(plt.Rectangle(**rect_params))
#     text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
#     ax.annotate(**text_params)
#     ax.axis('off')
#     ax.set_title("$M = $"+im_name, fontsize=21)

# fig.savefig(path_result+'/PnP_Prox_num_noise_influence.png')
# plt.show()


# path_result = "/beegfs/mrenaud/Result_Average_PnP/deblurring/set1c/PnP_Prox_k_0/noise_10.0/"

# lamb_list = [0.05, 0.075, 0.1, 0.125, 0.15] #0.01, 0.5

# sigma_list = [5.0, 10.0, 15.0, 18.0, 30.0, 50.0] #20.0

# n = len(lamb_list)
# m = len(sigma_list)

# #size of the black rectangle
# height = 22
# width = 190

# fig = plt.figure(figsize = (m*5, n*5))
# for i in range(n):
#     for j in range(m):
#         dic_PnP = np.load(path_result + "lamb_"+str(lamb_list[i])+"/sigma_denoiser_"+str(sigma_list[j])+"/dict_0_results.npy", allow_pickle=True).item()

#         # gt = dic_PnP["GT"][:256,:256]
#         # deblur_PnP = (dic_PnP["Deblur"][:256,:256], dic_PnP["PSNR_output"], dic_PnP["SSIM_output"], dic_PnP["LPIPS_output"], dic_PnP["BRISQUE_output"])
#         # blur = (dic_PnP["Blur"][:256,:256], dic_PnP["PSNR_blur"], dic_PnP["SSIM_blur"], dic_PnP["LPIPS_blur"], dic_PnP["BRISQUE_blur"])
#         # k = dic_PnP["kernel"]
#         PnP = (dic_PnP["Deblur"][:256,:256], dic_PnP["PSNR_output"], dic_PnP["SSIM_output"], dic_PnP["LPIPS_output"], dic_PnP["BRISQUE_output"])
        
#         im = PnP
#         ax = fig.add_subplot(n,m,1+j+m*i)
#         ax.imshow(im[0])
#         rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
#         ax.add_patch(plt.Rectangle(**rect_params))
#         text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
#         ax.annotate(**text_params)
#         ax.axis('off')
#         ax.set_title("$\lambda = {}, \sigma = {}$".format(lamb_list[i], sigma_list[j]), fontsize=21)
            
#     fig.savefig(path_result+'/All_results_lamb_sigma_PnP_Prox.png')
#     plt.show()
