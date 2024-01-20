import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
from utils.utils_restoration import rescale, psnr, array2tensor, tensor2array, get_gaussian_noise_parameters, create_out_dir, single2uint,crop_center, matlab_style_gauss2D, imread_uint, imsave
from argparse import ArgumentParser
import os
import argparse
import cv2
import imageio

parser = ArgumentParser()
parser.add_argument('--fig_number', type=int, default=-1)
parser.add_argument('--table_number', type=int, default=0)
pars = parser.parse_args()

path_figure = "/beegfs/mrenaud/Result_Average_PnP/figure/"

if pars.fig_number == 0:
    #generate figure for deblurring in the paper.
    path_result = "/beegfs/mrenaud/Result_Average_PnP/deblurring/set4c/"

    # name_list = ["Average_PnP_k_0/noise_10.0","PnP_Prox_k_0/noise_10.0"]
    im_name_list = ["0"]#, "1"]#, "2", "3"]

    im_list = []

    name_fig_list = ["Observation", "RED", "Average PnP"]#, "Average PnP Prox", "PnP GD"

    n = 2
    m = 2

    #size of the black rectangle
    height = 35
    width = 150
    indices = [(0,0), (0, 1), (1,0), (1,1)]
    fig = plt.figure(figsize = (15, 30))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.7])

    text_size = 30

    for i, im_name in enumerate(im_name_list):
        width = 270
        dic_PnPProx = np.load(path_result + "PnP_Prox_k_0/noise_10.0/dict_"+im_name+"_results.npy", allow_pickle=True).item()
        # dic_PnPGD = np.load(path_result + "PnP_GD_k_0/noise_10.0/dict_"+im_name+"_results.npy", allow_pickle=True).item()
        dic_APnP = np.load(path_result + "Average_PnP_k_0/noise_10.0/annealing_number_16/dict_"+im_name+"_results.npy", allow_pickle=True).item()
        # dic_APnPProx = np.load(path_result + "Average_PnP_Prox_k_0/noise_10.0/dict_"+im_name+"_results.npy", allow_pickle=True).item()

        gt = dic_PnPProx["GT"]
        deblur_PnPProx = (dic_PnPProx["Deblur"], dic_PnPProx["PSNR_output"], dic_PnPProx["SSIM_output"], dic_PnPProx["LPIPS_output"], dic_PnPProx["BRISQUE_output"])
        # deblur_PnPGD = (dic_PnPGD["Deblur"], dic_PnPGD["PSNR_output"], dic_PnPGD["SSIM_output"], dic_PnPGD["LPIPS_output"], dic_PnPGD["BRISQUE_output"])
        blur = (dic_PnPProx["Blur"], dic_PnPProx["PSNR_blur"], dic_PnPProx["SSIM_blur"], dic_PnPProx["LPIPS_blur"], dic_PnPProx["BRISQUE_blur"])
        k = dic_PnPProx["kernel"]
        deblur_APnP = (dic_APnP["Deblur"], dic_APnP["PSNR_output"], dic_APnP["SSIM_output"], dic_APnP["LPIPS_output"], dic_APnP["BRISQUE_output"])
        # deblur_APnPProx = (dic_APnPProx["Deblur"], dic_APnPProx["PSNR_output"], dic_APnPProx["SSIM_output"], dic_APnPProx["LPIPS_output"], dic_APnPProx["BRISQUE_output"])
        F_list = dic_APnP['F_list']

        c = 140
        wid, hei = 70, 70
        x_c, y_c = 230, 150
        ax = plt.subplot(gs[indices[0]])

        #add a zoom of the image
        patch_c = cv2.resize(gt[y_c:y_c+hei, x_c:x_c+wid], dsize =(c,c), interpolation=cv2.INTER_CUBIC)
        gt[-patch_c.shape[0]:,-patch_c.shape[1]:] = patch_c
        rect_params_z = {'xy': (gt.shape[1]-patch_c.shape[1]-1, gt.shape[0]-patch_c.shape[0]-1), 'width': c, 'height': c, 'linewidth': 2, 'edgecolor': 'red', 'facecolor': 'none'}
        ax.add_patch(plt.Rectangle(**rect_params_z))

        ax.imshow(gt)
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': r"PSNR$\uparrow$BRISQUE$\downarrow$", 'color': 'white', 'fontsize': text_size, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        
        #add a color rectangle
        rect_params_c = {'xy': (x_c, y_c), 'width': wid, 'height': hei, 'linewidth': 2, 'edgecolor': 'red', 'facecolor': 'none'}
        ax.add_patch(plt.Rectangle(**rect_params_c))

        ax.axis('off')
        ax.set_title("Ground Truth", fontsize=text_size)

        width = 180

        for j, im in enumerate([blur, deblur_PnPProx, deblur_APnP]):
            ax = plt.subplot(gs[indices[1+j]])
            if j ==0:
                k_resize = cv2.resize(k, dsize =(c,c), interpolation=cv2.INTER_CUBIC)
                im[0][-k_resize.shape[0]:,:k_resize.shape[1]] = k_resize[:,:,None]*np.ones(3)[None,None,:] / np.max(k_resize)
            
            #add a zoom of the image
            patch_c = cv2.resize(im[0][y_c:y_c+hei, x_c:x_c+wid], dsize =(c,c), interpolation=cv2.INTER_CUBIC)
            im[0][-patch_c.shape[0]:,-patch_c.shape[1]:] = patch_c
            rect_params_z = {'xy': (im[0].shape[1]-patch_c.shape[1]-1, im[0].shape[0]-patch_c.shape[0]-1), 'width': c, 'height': c, 'linewidth': 2, 'edgecolor': 'red', 'facecolor': 'none'}
            ax.add_patch(plt.Rectangle(**rect_params_z))
            
            ax.imshow(im[0])
            rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
            ax.add_patch(plt.Rectangle(**rect_params))
            text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}".format(im[1], im[4]), 'color': 'white', 'fontsize': text_size, 'va': 'top', 'ha': 'left'}
            ax.annotate(**text_params)

            #add a color rectangle
            rect_params_c = {'xy': (x_c, y_c), 'width': wid, 'height': hei, 'linewidth': 2, 'edgecolor': 'red', 'facecolor': 'none'}
            ax.add_patch(plt.Rectangle(**rect_params_c))

            ax.axis('off')
            ax.set_title(name_fig_list[j], fontsize=text_size)

        ax = plt.subplot(gs[2, :])
        # ax = fig.add_subplot(m,n,5)
        ax.plot(np.arange(1200, 1500),F_list[-300:])
        ax.axis('on')
        ax.set_yticks([546, 550])
        ax.yaxis.set_tick_params(labelsize=text_size)
        ax.set_xticks([1200, 1500])
        ax.xaxis.set_tick_params(labelsize=text_size)
        ax.set_title(r"$\mathcal{F}(\mathbf{x_k}, \mathbf{y}) + \frac{\alpha}{\sigma^2} g_{\sigma}(\mathbf{x_k})$", fontsize=text_size)

    fig.savefig(path_figure+'/Results_restoration_deblurring.png')
    plt.show()


if pars.fig_number == 1:
    # generate figure for deblurring in the Annexe. With the decrease of the function for the indian image.
    path_result = "/beegfs/mrenaud/Result_Average_PnP/deblurring/set4c/"

    name_fig_list = ["Observation", "Average PnP"]

    #size of the black rectangle
    height = 25
    width = 150
    fig = plt.figure(figsize = (15, 24))
    gs = gridspec.GridSpec(3, 2, width_ratios = [1, 2])

    text_size = 30
    text_size_label = 14

    width = 350
    # dic_PnPProx = np.load(path_result + "PnP_Prox_k_0/noise_10.0/dict_1_results.npy", allow_pickle=True).item()
    # dic_PnPGD = np.load(path_result + "PnP_GD_k_0/noise_10.0/dict_"+im_name+"_results.npy", allow_pickle=True).item()
    dic_APnP = np.load(path_result + "Average_PnP_k_0/noise_5.0/annealing_number_16/dict_0_results.npy", allow_pickle=True).item()
    # dic_APnPProx = np.load(path_result + "Average_PnP_Prox_k_0/noise_10.0/dict_"+im_name+"_results.npy", allow_pickle=True).item()

    gt = dic_APnP["GT"]
    # deblur_PnPProx = (dic_PnPProx["Deblur"], dic_PnPProx["PSNR_output"], dic_PnPProx["SSIM_output"], dic_PnPProx["LPIPS_output"], dic_PnPProx["BRISQUE_output"])
    # deblur_PnPGD = (dic_PnPGD["Deblur"], dic_PnPGD["PSNR_output"], dic_PnPGD["SSIM_output"], dic_PnPGD["LPIPS_output"], dic_PnPGD["BRISQUE_output"])
    blur = (dic_APnP["Blur"], dic_APnP["PSNR_blur"], dic_APnP["SSIM_blur"], dic_APnP["LPIPS_blur"], dic_APnP["BRISQUE_blur"])
    k = dic_APnP["kernel"]
    deblur_APnP = (dic_APnP["Deblur"], dic_APnP["PSNR_output"], dic_APnP["SSIM_output"], dic_APnP["LPIPS_output"], dic_APnP["BRISQUE_output"])
    # deblur_APnPProx = (dic_APnPProx["Deblur"], dic_APnPProx["PSNR_output"], dic_APnPProx["SSIM_output"], dic_APnPProx["LPIPS_output"], dic_APnPProx["BRISQUE_output"])
    F_list = dic_APnP['F_list']
    lamb_list = dic_APnP['lamb_tab']
    std_list = dic_APnP['std_tab']

    ax = plt.subplot(gs[0, 0])
    ax.imshow(gt)
    rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
    ax.add_patch(plt.Rectangle(**rect_params))
    text_params = {'xy': (5, 5), 'text': r"PSNR$\uparrow$SSIM$\uparrow$LPIPS$\downarrow$BRISQUE$\downarrow$", 'color': 'white', 'fontsize': text_size_label, 'va': 'top', 'ha': 'left'}
    ax.annotate(**text_params)
    ax.axis('off')
    ax.set_title("Ground Truth", fontsize=text_size)

    width = 210

    for j, im in enumerate([blur, deblur_APnP]):
        ax = plt.subplot(gs[1+j, 0])
        if j ==0:
            c = 110
            k_resize = cv2.resize(k, dsize =(c,c), interpolation=cv2.INTER_CUBIC)
            im[0][-k_resize.shape[0]:,:k_resize.shape[1]] = k_resize[:,:,None]*np.ones(3)[None,None,:] / np.max(k_resize)
        ax.imshow(im[0])
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': text_size_label, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
        ax.set_title(name_fig_list[j], fontsize=text_size)

    ax = plt.subplot(gs[0, 1])
    # ax = fig.add_subplot(m,n,5)
    ax.plot(np.arange(1500),lamb_list)
    ax.axis('on')
    ax.set_yticks([0.1, 1.])
    ax.yaxis.set_tick_params(labelsize=text_size)
    ax.set_xticks([0, 1500])
    ax.xaxis.set_tick_params(labelsize=text_size)
    ax.set_title(r"$\alpha_k$", fontsize=text_size)

    ax = plt.subplot(gs[1, 1])
    # ax = fig.add_subplot(m,n,5)
    ax.plot(np.arange(1500),std_list)
    ax.axis('on')
    ax.set_yticks([0.00980, 0.03529], labels = [r"$\frac{2.5}{255}$", r"$\frac{9}{255}$"])
    ax.yaxis.set_tick_params(labelsize=text_size)
    ax.set_xticks([0, 1500])
    ax.xaxis.set_tick_params(labelsize=text_size)
    ax.set_title(r"$\sigma_k$", fontsize=text_size)

    ax = plt.subplot(gs[2, 1])
    # ax = fig.add_subplot(m,n,5)
    ax.plot(np.arange(1500),F_list)
    ax.axis('on')
    ax.set_yticks([150, 275])
    ax.yaxis.set_tick_params(labelsize=text_size)
    ax.set_xticks([0, 1500])
    ax.xaxis.set_tick_params(labelsize=text_size)
    ax.set_title(r"$\mathcal{F}(\mathbf{x_k}, \mathbf{y}) + \alpha_k \mathcal{R}_{\sigma_k}(\mathbf{x_k})$", fontsize=text_size)

    fig.savefig(path_figure+'Results_restoration_annexe_deblurring_parameters.png')
    plt.show()



if pars.fig_number == 2:
    # generate figure for deblurring in the Annexe. With the decrease of the function for the indian image.
    path_result = "/beegfs/mrenaud/Result_Average_PnP/deblurring/set4c/"

    name_fig_list = ["Observation", "Average PnP"]

    #size of the black rectangle
    height = 25
    width = 150
    fig = plt.figure(figsize = (15, 24))
    gs = gridspec.GridSpec(3, 2, width_ratios = [1, 2])

    text_size = 30
    text_size_label = 14

    width = 350
    # dic_PnPProx = np.load(path_result + "PnP_Prox_k_0/noise_10.0/dict_1_results.npy", allow_pickle=True).item()
    # dic_PnPGD = np.load(path_result + "PnP_GD_k_0/noise_10.0/dict_"+im_name+"_results.npy", allow_pickle=True).item()
    dic_APnP = np.load(path_result + "Average_PnP_k_0/noise_5.0/annealing_number_16/dict_1_results.npy", allow_pickle=True).item()
    # dic_APnPProx = np.load(path_result + "Average_PnP_Prox_k_0/noise_10.0/dict_"+im_name+"_results.npy", allow_pickle=True).item()

    gt = dic_APnP["GT"]
    # deblur_PnPProx = (dic_PnPProx["Deblur"], dic_PnPProx["PSNR_output"], dic_PnPProx["SSIM_output"], dic_PnPProx["LPIPS_output"], dic_PnPProx["BRISQUE_output"])
    # deblur_PnPGD = (dic_PnPGD["Deblur"], dic_PnPGD["PSNR_output"], dic_PnPGD["SSIM_output"], dic_PnPGD["LPIPS_output"], dic_PnPGD["BRISQUE_output"])
    blur = (dic_APnP["Blur"], dic_APnP["PSNR_blur"], dic_APnP["SSIM_blur"], dic_APnP["LPIPS_blur"], dic_APnP["BRISQUE_blur"])
    k = dic_APnP["kernel"]
    deblur_APnP = (dic_APnP["Deblur"], dic_APnP["PSNR_output"], dic_APnP["SSIM_output"], dic_APnP["LPIPS_output"], dic_APnP["BRISQUE_output"])
    # deblur_APnPProx = (dic_APnPProx["Deblur"], dic_APnPProx["PSNR_output"], dic_APnPProx["SSIM_output"], dic_APnPProx["LPIPS_output"], dic_APnPProx["BRISQUE_output"])
    F_list = dic_APnP['F_list']
    lamb_list = dic_APnP['lamb_tab']
    std_list = dic_APnP['std_tab']

    ax = plt.subplot(gs[0, 0])
    ax.imshow(gt)
    rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
    ax.add_patch(plt.Rectangle(**rect_params))
    text_params = {'xy': (5, 5), 'text': r"PSNR$\uparrow$SSIM$\uparrow$LPIPS$\downarrow$BRISQUE$\downarrow$", 'color': 'white', 'fontsize': text_size_label, 'va': 'top', 'ha': 'left'}
    ax.annotate(**text_params)
    ax.axis('off')
    ax.set_title("Ground Truth", fontsize=text_size)

    width = 210

    for j, im in enumerate([blur, deblur_APnP]):
        ax = plt.subplot(gs[1+j, 0])
        if j ==0:
            c = 110
            k_resize = cv2.resize(k, dsize =(c,c), interpolation=cv2.INTER_CUBIC)
            im[0][-k_resize.shape[0]:,:k_resize.shape[1]] = k_resize[:,:,None]*np.ones(3)[None,None,:] / np.max(k_resize)
        ax.imshow(im[0])
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': text_size_label, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
        ax.set_title(name_fig_list[j], fontsize=text_size)

    ax = plt.subplot(gs[0, 1])
    # ax = fig.add_subplot(m,n,5)
    ax.plot(np.arange(1500),lamb_list)
    ax.axis('on')
    ax.set_yticks([0.1, 1.])
    ax.yaxis.set_tick_params(labelsize=text_size)
    ax.set_xticks([0, 1500])
    ax.xaxis.set_tick_params(labelsize=text_size)
    ax.set_title(r"$\alpha_k$", fontsize=text_size)

    ax = plt.subplot(gs[1, 1])
    # ax = fig.add_subplot(m,n,5)
    ax.plot(np.arange(1500),std_list)
    ax.axis('on')
    ax.set_yticks([0.00980, 0.03529], labels = [r"$\frac{2.5}{255}$", r"$\frac{9}{255}$"])
    ax.yaxis.set_tick_params(labelsize=text_size)
    ax.set_xticks([0, 1500])
    ax.xaxis.set_tick_params(labelsize=text_size)
    ax.set_title(r"$\sigma_k$", fontsize=text_size)

    ax = plt.subplot(gs[2, 1])
    # ax = fig.add_subplot(m,n,5)
    ax.plot(np.arange(1500),F_list)
    ax.axis('on')
    ax.set_yticks([150, 275])
    ax.yaxis.set_tick_params(labelsize=text_size)
    ax.set_xticks([0, 1500])
    ax.xaxis.set_tick_params(labelsize=text_size)
    ax.set_title(r"$\mathcal{F}(\mathbf{x_k}, \mathbf{y}) + \alpha_k \mathcal{R}_{\sigma_k}(\mathbf{x_k})$", fontsize=text_size)

    fig.savefig(path_figure+'Results_restoration_annexe_deblurring_parameters_2.png')
    plt.show()



if pars.fig_number == 3:
    # Show the influence of a last denoising on the result
    path_result = "/beegfs/mrenaud/Result_Average_PnP/deblurring/set4c/Average_PnP_k_0/noise_10.0/annealing_number_16/"

    # name_list = ["Average_kernel_0","kernel_0"]
    im_name_list = ["0", "1", "2", "3"]
    # ["0.05", "0.1", "0.3", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "5.0", "10.0", "20.0"]

    im_list = []

    # name_fig_list = ["Observation", "PnP Prox", "Average PnP"]

    m = len(im_name_list)
    n = 4

    #size of the black rectangle
    height = 22
    width = 280

    fig = plt.figure(figsize = (n*5, m*8))
    for i, im_name in enumerate(im_name_list):
        dic_AveragePnP = np.load(path_result + "dict_"+str(im_name)+"_results.npy", allow_pickle=True).item()

        gt = dic_AveragePnP["GT"]
        blur = (dic_AveragePnP["Blur"], dic_AveragePnP["PSNR_blur"], dic_AveragePnP["SSIM_blur"], dic_AveragePnP["LPIPS_blur"], dic_AveragePnP["BRISQUE_blur"])
        k = dic_AveragePnP["kernel"]

        ax = fig.add_subplot(m,n,1+4*i)
        ax.imshow(gt)
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "PSNR/SSIM/LPIPS/BRISQUE", 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
        ax.set_title("Ground Truth", fontsize=21)

        width = 230
        im = blur
        ax = fig.add_subplot(m,n,2+4*i)
        c = 50
        k_resize = cv2.resize(k, dsize =(c,c), interpolation=cv2.INTER_CUBIC)
        im[0][-k_resize.shape[0]:,:k_resize.shape[1]] = k_resize[:,:,None]*np.ones(3)[None,None,:] / np.max(k_resize)
        ax.imshow(im[0])
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
        ax.set_title("Observation", fontsize=21)

        list_name = ["Deblur", "Deblur + Denoised"]
        deblur_APnP = (dic_AveragePnP["Deblur"], dic_AveragePnP["PSNR_output"], dic_AveragePnP["SSIM_output"], dic_AveragePnP["LPIPS_output"], dic_AveragePnP["BRISQUE_output"])
        deblur_den_APnP = (dic_AveragePnP["output_den_img"], dic_AveragePnP["output_den_psnr"], dic_AveragePnP["output_den_ssim"], dic_AveragePnP["output_den_lpips"], dic_AveragePnP["output_den_brisque"])
        plt.imsave(path_result+"images/img_"+im_name+"_deblur_denoised.png", single2uint(np.clip(dic_AveragePnP["output_den_img"], 0, 1)))
        for j, im in enumerate([deblur_APnP, deblur_den_APnP]):
            ax = fig.add_subplot(m,n,3+4*i+j)
            ax.imshow(im[0])
            rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
            ax.add_patch(plt.Rectangle(**rect_params))
            text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
            ax.annotate(**text_params)
            ax.axis('off')
            ax.set_title(list_name[j], fontsize=21)

    fig.savefig(path_figure+'/Restoration_deblur_and_denoised.png')
    plt.show()


if pars.fig_number == 4:
    # Image a uncertainity to the seed of the randomness for Average PnP algorithm.

    path_result = "/beegfs/mrenaud/Result_Average_PnP/deblurring/set1c/Average_PnP_k_0/noise_5.0/"

    # name_list = ["Average_kernel_0","kernel_0"]
    p = 10
    im_name_list = [str(i) for i in range(1,p)]
    # ["0.05", "0.1", "0.3", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "5.0", "10.0", "20.0"]

    # name_fig_list = ["Observation", "PnP Prox", "Average PnP"]

    m = 1
    n = 4

    #size of the black rectangle
    height = 22


    im_list = []
    im_list_2 = []

    fig = plt.figure(figsize = (n*5, m*8))
    for i, im_name in enumerate(im_name_list):
        dic_AveragePnP = np.load(path_result + "seed_"+str(im_name)+"/annealing_number_16/dict_0_results.npy", allow_pickle=True).item()
        if i ==0:
            gt = dic_AveragePnP["GT"]
            im_list.append(gt)
            blur = (dic_AveragePnP["Blur"], dic_AveragePnP["PSNR_blur"], dic_AveragePnP["SSIM_blur"], dic_AveragePnP["LPIPS_blur"], dic_AveragePnP["BRISQUE_blur"])
            im_list.append(blur)
            k = dic_AveragePnP["kernel"]
        
        deblur_APnP = (dic_AveragePnP["Deblur"], dic_AveragePnP["PSNR_output"], dic_AveragePnP["SSIM_output"], dic_AveragePnP["LPIPS_output"], dic_AveragePnP["BRISQUE_output"])
        if i == 0:
            im_list.append(deblur_APnP)
        im_list_2.append(dic_AveragePnP["Deblur"])

    im_list_2 = np.array(im_list_2)
    print(im_list_2.shape)
    im_std = np.std(im_list_2, axis = 0)
    im_list.append(np.abs(1/np.log(im_std)) / np.max(np.abs(1/np.log(im_std))))

    # for i in range(20):
    #     im_list_3 = im_list_2[:i, :, :, :]
    #     im_std_2 = np.std(im_list_3, axis = 0)
    #     print(np.sum(im_std_2))


    for i, im in enumerate(im_list):
        if i == 0 or i == 3:
            width = 270
            ax = fig.add_subplot(m,n,1+i)
            if i == 0:
                ax.imshow(im)
                ax.set_title("Ground Truth", fontsize=21)
                rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
                ax.add_patch(plt.Rectangle(**rect_params))
                text_params = {'xy': (5, 5), 'text': r"PSNR$\uparrow$LPIPS$\downarrow$BRISQUE$\downarrow$", 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
                ax.annotate(**text_params)  
            if i == 3:
                ax.imshow(im)
                ax.set_title("Standard deviation, max = {:.2f}".format(np.max(im_std)), fontsize = 21)
            ax.axis('off')
            
        else:
            width = 170
            ax = fig.add_subplot(m,n,1+i)
            if i==1:
                c = 100
                k_resize = cv2.resize(k, dsize =(c,c), interpolation=cv2.INTER_CUBIC)
                im[0][-k_resize.shape[0]:,:k_resize.shape[1]] = k_resize[:,:,None]*np.ones(3)[None,None,:] / np.max(k_resize)
            ax.imshow(im[0])
            rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
            ax.add_patch(plt.Rectangle(**rect_params))
            text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
            ax.annotate(**text_params)
            ax.axis('off')
            if i == 1:
                ax.set_title("Observation", fontsize=21)
            if i == 2:
                ax.set_title("Average PnP", fontsize=21)

    fig.savefig(path_figure+'/Restoration_deblurring_seed_sensitivity.png')
    plt.show()



if pars.fig_number == 5:
    # Generate figure for inpainting results.
    path_result = "/beegfs/mrenaud/Result_Average_PnP/inpainting/set1c/"

    name_fig_list = ["Observation", "Average PnP", "RED"]

    #size of the black rectangle
    height = 25
    width = 150
    fig = plt.figure(figsize = (15, 24))
    gs = gridspec.GridSpec(2, 2)

    prop = "0.5"
    text_size = 30
    text_size_label = 21

    width = 350
    dic_APnP = np.load(path_result + "Average_PnP_Prox/noise_0/mask_prop_"+prop+"/annealing_number_16/dict_0_results.npy", allow_pickle=True).item()

    gt = dic_APnP["GT"]
    blur = (dic_APnP["Masked"], dic_APnP["PSNR_masked"], dic_APnP["SSIM_masked"], dic_APnP["LPIPS_masked"], dic_APnP["BRISQUE_masked"])
    deblur_APnP = (dic_APnP["Inpainted"], dic_APnP["PSNR_output"], dic_APnP["SSIM_output"], dic_APnP["LPIPS_output"], dic_APnP["BRISQUE_output"])

    dic_PnP = np.load(path_result + "PnP_Prox/noise_0/mask_prop_"+prop+"/annealing_number_16/dict_0_results.npy", allow_pickle=True).item()
    deblur_PnP = (dic_PnP["Inpainted"], dic_PnP["PSNR_output"], dic_PnP["SSIM_output"], dic_PnP["LPIPS_output"], dic_PnP["BRISQUE_output"])

    ax = plt.subplot(gs[0, 0])
    ax.imshow(gt)
    rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
    ax.add_patch(plt.Rectangle(**rect_params))
    text_params = {'xy': (5, 5), 'text': r"PSNR$\uparrow$SSIM$\uparrow$LPIPS$\downarrow$BRISQUE$\downarrow$", 'color': 'white', 'fontsize': text_size_label, 'va': 'top', 'ha': 'left'}
    ax.annotate(**text_params)
    ax.axis('off')
    ax.set_title("Ground Truth", fontsize=text_size)

    width = 210
    index = [(0,1), (1,0), (1,1)]
    for j, im in enumerate([blur, deblur_APnP, deblur_PnP]):
        ax = plt.subplot(gs[index[j]])
        ax.imshow(im[0])
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': text_size_label, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
        ax.set_title(name_fig_list[j], fontsize=text_size)

    fig.savefig(path_figure+'Results_restoration_inpainting.png')
    plt.show()

if pars.fig_number == 6:
    # Generate figure for inpainting results.
    path_result = "/beegfs/mrenaud/Result_Average_PnP/inpainting/set1c/"

    name_fig_list = ["Observation", "Average PnP", "RED"]

    #size of the black rectangle
    height = 25
    width = 150
    fig = plt.figure(figsize = (15, 24))
    gs = gridspec.GridSpec(2, 2)

    prop = "0.2"
    text_size = 30
    text_size_label = 21

    width = 350
    dic_APnP = np.load(path_result + "Average_PnP_Prox/noise_0/mask_prop_"+prop+"/annealing_number_16/dict_0_results.npy", allow_pickle=True).item()

    gt = dic_APnP["GT"]
    blur = (dic_APnP["Masked"], dic_APnP["PSNR_masked"], dic_APnP["SSIM_masked"], dic_APnP["LPIPS_masked"], dic_APnP["BRISQUE_masked"])
    deblur_APnP = (dic_APnP["Inpainted"], dic_APnP["PSNR_output"], dic_APnP["SSIM_output"], dic_APnP["LPIPS_output"], dic_APnP["BRISQUE_output"])

    dic_PnP = np.load(path_result + "PnP_Prox/noise_0/mask_prop_"+prop+"/annealing_number_16/dict_0_results.npy", allow_pickle=True).item()
    deblur_PnP = (dic_PnP["Inpainted"], dic_PnP["PSNR_output"], dic_PnP["SSIM_output"], dic_PnP["LPIPS_output"], dic_PnP["BRISQUE_output"])

    ax = plt.subplot(gs[0, 0])
    ax.imshow(gt)
    rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
    ax.add_patch(plt.Rectangle(**rect_params))
    text_params = {'xy': (5, 5), 'text': r"PSNR$\uparrow$SSIM$\uparrow$LPIPS$\downarrow$BRISQUE$\downarrow$", 'color': 'white', 'fontsize': text_size_label, 'va': 'top', 'ha': 'left'}
    ax.annotate(**text_params)
    ax.axis('off')
    ax.set_title("Ground Truth", fontsize=text_size)

    width = 210
    index = [(0,1), (1,0), (1,1)]
    for j, im in enumerate([blur, deblur_APnP, deblur_PnP]):
        ax = plt.subplot(gs[index[j]])
        ax.imshow(im[0])
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': text_size_label, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
        ax.set_title(name_fig_list[j], fontsize=text_size)

    fig.savefig(path_figure+'Results_restoration_inpainting_2.png')
    plt.show()


if pars.fig_number == 7:
    # Generate a figure with various lanmbda and sigma for RED
    path_result = "/beegfs/mrenaud/Result_Average_PnP/deblurring/set1c/PnP_Prox_k_0/noise_10.0/"

    lamb_list = [0.05, 0.075, 0.1, 0.125, 0.15] #0.01, 0.5
    sigma_list = [5.0, 10.0, 15.0, 18.0, 30.0, 50.0] #20.0

    n = len(lamb_list)
    m = len(sigma_list)

    #size of the black rectangle
    height = 22
    width = 190

    fig = plt.figure(figsize = (m*5, n*5))
    for i in range(n):
        for j in range(m):
            dic_PnP = np.load(path_result + "lamb_"+str(lamb_list[i])+"/sigma_denoiser_"+str(sigma_list[j])+"/dict_0_results.npy", allow_pickle=True).item()

            # gt = dic_PnP["GT"][:256,:256]
            # deblur_PnP = (dic_PnP["Deblur"][:256,:256], dic_PnP["PSNR_output"], dic_PnP["SSIM_output"], dic_PnP["LPIPS_output"], dic_PnP["BRISQUE_output"])
            # blur = (dic_PnP["Blur"][:256,:256], dic_PnP["PSNR_blur"], dic_PnP["SSIM_blur"], dic_PnP["LPIPS_blur"], dic_PnP["BRISQUE_blur"])
            # k = dic_PnP["kernel"]
            PnP = (dic_PnP["Deblur"][:256,:256], dic_PnP["PSNR_output"], dic_PnP["SSIM_output"], dic_PnP["LPIPS_output"], dic_PnP["BRISQUE_output"])
            
            im = PnP
            ax = fig.add_subplot(n,m,1+j+m*i)
            ax.imshow(im[0])
            rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
            ax.add_patch(plt.Rectangle(**rect_params))
            text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
            ax.annotate(**text_params)
            ax.axis('off')
            ax.set_title("$\lambda = {}, \sigma = {}$".format(lamb_list[i], sigma_list[j]), fontsize=21)
                
        fig.savefig(path_figure+'/All_results_lamb_sigma_PnP_Prox.png')
        plt.show()


if pars.table_number == 1:
    #generate the result of deblurring for table of result on CBSD10 dataset for 3 differents level of noise with various restoration methods.

    path_result = "/beegfs/mrenaud/Result_Average_PnP/deblurring/CBSD10/"

    noise_list = ["5.0", "10.0", "20.0"]

    n = 10
    m = 10
    for noise in noise_list:
        print("Noise level :", noise)
        output_psnr = [[],[]]
        output_ssim = [[],[]]
        output_lpips = [[],[]]
        output_brisque = [[],[]]
        for i in tqdm(range(m)):
            for j in range(n):
                dic_RED = np.load(path_result + "PnP_Prox_k_"+str(i)+"/noise_"+noise+"/annealing_number_16/dict_"+str(j)+"_results.npy", allow_pickle=True).item()
                output_psnr[0].append(dic_RED["PSNR_output"])
                output_ssim[0].append(dic_RED["SSIM_output"])
                output_lpips[0].append(dic_RED["LPIPS_output"])
                output_brisque[0].append(dic_RED["BRISQUE_output"])
                dic_APnP = np.load(path_result + "Average_PnP_k_"+str(i)+"/noise_"+noise+"/annealing_number_16/dict_"+str(j)+"_results.npy", allow_pickle=True).item()
                output_psnr[1].append(dic_APnP["PSNR_output"])
                output_ssim[1].append(dic_APnP["SSIM_output"])
                output_lpips[1].append(dic_APnP["LPIPS_output"])
                output_brisque[1].append(dic_APnP["BRISQUE_output"])
    
        output_psnr = np.array(output_psnr)
        output_ssim = np.array(output_ssim)
        output_lpips = np.array(output_lpips)
        output_brisque = np.array(output_brisque)
        print("RED PSNR/SSIM/LPIPS/BRISQUE : {:.2f} & {:.2f} & {:.2f} & {:.2f}".format(np.mean(output_psnr[0]),np.mean(output_ssim[0]),np.mean(output_lpips[0]),np.mean(output_brisque[0])))
        print("Average PnP PSNR/SSIM/LPIPS/BRISQUE : {:.2f} & {:.2f} & {:.2f} & {:.2f}".format(np.mean(output_psnr[1]),np.mean(output_ssim[1]),np.mean(output_lpips[1]),np.mean(output_brisque[1])))


if pars.table_number == 2:
    #generate the result of inpainting for table of result on CBSD68 dataset.

    path_result = "/beegfs/mrenaud/Result_Average_PnP/inpainting/CBSD68/"

    n = 68
    output_psnr = [[],[]]
    output_ssim = [[],[]]
    output_lpips = [[],[]]
    output_brisque = [[],[]]
    for i in tqdm(range(n)):
        dic_RED = np.load(path_result + "PnP_Prox/noise_0/mask_prop_0.5/annealing_number_16/dict_"+str(i)+"_results.npy", allow_pickle=True).item()
        output_psnr[0].append(dic_RED["PSNR_output"])
        output_ssim[0].append(dic_RED["SSIM_output"])
        output_lpips[0].append(dic_RED["LPIPS_output"])
        output_brisque[0].append(dic_RED["BRISQUE_output"])
        dic_APnPProx = np.load(path_result + "Average_PnP_Prox/noise_0/mask_prop_0.5/annealing_number_16/dict_"+str(i)+"_results.npy", allow_pickle=True).item()
        output_psnr[1].append(dic_APnPProx["PSNR_output"])
        output_ssim[1].append(dic_APnPProx["SSIM_output"])
        output_lpips[1].append(dic_APnPProx["LPIPS_output"])
        output_brisque[1].append(dic_APnPProx["BRISQUE_output"])

    output_psnr = np.array(output_psnr)
    output_ssim = np.array(output_ssim)
    output_lpips = np.array(output_lpips)
    output_brisque = np.array(output_brisque)
    print("PnP_Prox PSNR/SSIM/LPIPS/BRISQUE : {:.2f} / {:.2f} / {:.2f} / {:.2f}".format(np.mean(output_psnr[0]),np.mean(output_ssim[0]),np.mean(output_lpips[0]),np.mean(output_brisque[0])))
    print("Average_PnP_Prox PSNR/SSIM/LPIPS/BRISQUE :{:.2f} / {:.2f} / {:.2f} / {:.2f}".format(np.mean(output_psnr[1]),np.mean(output_ssim[1]),np.mean(output_lpips[1]),np.mean(output_brisque[1])))


if pars.table_number == 3:
    #generate the result of inpainting for table of result of DiffPIR on CBSD10 dataset.

    path_result = "/beegfs/mrenaud/Result_Average_PnP/deblurring/CBSD10/DiffPIR"

    for noise in ["5.0", "10.0", "20.0"]:
        n = 10
        output_psnr = []
        output_ssim = []
        output_lpips = []
        output_brisque = []
        for i in tqdm(range(n)):
            for j in range(n):
                dic_Diff = np.load(path_result + "/kernel_"+str(i)+"/noise_level_"+noise+"/dict_"+str(j)+"_results.npy", allow_pickle=True).item()
                output_psnr.append(dic_Diff["PSNR_output"])
                output_ssim.append(dic_Diff["SSIM_output"])
                output_lpips.append(dic_Diff["LPIPS_output"])
                output_brisque.append(dic_Diff["BRISQUE_output"])

        output_psnr = np.array(output_psnr)
        output_ssim = np.array(output_ssim)
        output_lpips = np.array(output_lpips)
        output_brisque = np.array(output_brisque)
        print("Noise = "+noise)
        print("DiffPIR PSNR/SSIM/LPIPS/BRISQUE : {:.2f} & {:.2f} & {:.2f} & {:.2f}".format(np.mean(output_psnr),np.mean(output_ssim),np.mean(output_lpips),np.mean(output_brisque)))





if pars.fig_number == 8:
    #Generate a figure for the annealing number influence.
    path_result = "/beegfs/mrenaud/Result_Average_PnP/deblurring/set4c/Average_PnP_k_0/noise_10.0/"

    im_name_list = [1, 2, 3, 4, 11, 12, 13, 14]

    PSNR_list = []
    SSIM_list = []
    LPIPS_list = []
    BRISQUE_list = []
    im_list = []

    for i, im_name in enumerate(im_name_list):
        im_name = str(im_name)
        width = 280
        psnr, ssim, lpips, brisque = 0, 0, 0, 0
        im_blur_list = []
        im_GT_list = []
        im_j_list = []
        for j in range(4):
            dic = np.load(path_result + "annealing_number_"+im_name+"/dict_"+str(j)+"_results.npy", allow_pickle=True).item()
            if i==0:
                im_GT_list.append(dic['GT'])
                im_blur_list.append((dic["Blur"], dic["PSNR_blur"], dic["SSIM_blur"], dic["LPIPS_blur"], dic["BRISQUE_blur"]))
            psnr += dic['PSNR_output']
            ssim += dic['SSIM_output']
            lpips += dic['LPIPS_output']
            brisque += dic['BRISQUE_output']
            im_j_list.append((dic["Deblur"], dic["PSNR_output"], dic["SSIM_output"], dic["LPIPS_output"], dic["BRISQUE_output"]))
        if i == 0:
            im_list.append(im_GT_list)
            im_list.append(im_blur_list)
        im_list.append(im_j_list)
        PSNR_list.append(psnr/4); SSIM_list.append(ssim/4); LPIPS_list.append(lpips/4); BRISQUE_list.append(brisque/4)

    fig, ax = plt.subplots(1, 4, figsize = (20, 5))
    size_text = 25
    size_line = 5

    ax_i = ax[0]
    ax_i.plot(im_name_list, PSNR_list, linewidth = size_line)
    ax_i.scatter(im_name_list, PSNR_list, linewidth = size_line)
    ax_i.set_xlabel('$m$', fontsize = size_text)
    ax_i.xaxis.set_label_coords(0.5, -0.05)
    ax_i.set_xticks([1, 14])
    ax_i.xaxis.set_tick_params(labelsize=size_text)
    ax_i.set_yticks([18, 26])
    ax_i.yaxis.set_tick_params(labelsize=size_text)
    ax_i.set_title("PSNR", fontsize = size_text)

    ax_i = ax[1]
    ax_i.plot(im_name_list, SSIM_list, linewidth = size_line)
    ax_i.scatter(im_name_list, SSIM_list, linewidth = size_line)
    ax_i.set_xlabel('$m$', fontsize = size_text)
    ax_i.xaxis.set_label_coords(0.5, -0.05)
    ax_i.set_xticks([1, 14])
    ax_i.xaxis.set_tick_params(labelsize=size_text)
    ax_i.set_yticks([0.3, 0.7])
    ax_i.yaxis.set_tick_params(labelsize=size_text)
    ax_i.set_title("SSIM", fontsize = size_text)

    ax_i = ax[2]
    ax_i.plot(im_name_list, LPIPS_list, linewidth = size_line)
    ax_i.scatter(im_name_list, LPIPS_list, linewidth = size_line)
    ax_i.set_xlabel('$m$', fontsize = size_text)
    ax_i.xaxis.set_label_coords(0.5, -0.05)
    ax_i.set_xticks([1, 14])
    ax_i.xaxis.set_tick_params(labelsize=size_text)
    ax_i.set_yticks([0.2, 0.5])
    ax_i.yaxis.set_tick_params(labelsize=size_text)
    ax_i.set_title("LPIPS", fontsize = size_text)

    ax_i = ax[3]
    ax_i.plot(im_name_list, BRISQUE_list, linewidth = size_line)
    ax_i.scatter(im_name_list, BRISQUE_list, linewidth = size_line)
    ax_i.set_xlabel('$m$', fontsize = size_text)
    ax_i.xaxis.set_label_coords(0.5, -0.05)
    ax_i.set_xticks([1, 14])
    ax_i.xaxis.set_tick_params(labelsize=size_text)
    ax_i.set_yticks([10, 60])
    ax_i.yaxis.set_tick_params(labelsize=size_text)
    ax_i.set_title("BRISQUE", fontsize = size_text)

    fig.savefig(path_figure+'/Annealing_influence.png')

    #generate a figure for the annealing number influence on images.
    im_name_list = ["1", "4", "11", "16"]

    n = 2 #number of images 
    m = 2 + len(im_name_list) #number of reconstructions

    #size of the black rectangle
    height = 25
    c = 140
    wid, hei = 70, 70
    

    fig = plt.figure(figsize = (m*5, n*8))
    for i in range(n):
        if i == 0:
            x_c, y_c = 150, 150
        if i == 1:
            x_c, y_c = 100, 300
        for j in range(m):
            ax = fig.add_subplot(n,m,1+j+m*i)
            im = im_list[j][i]

            if j==0:
                #add a zoom of the image
                patch_c = cv2.resize(im[y_c:y_c+hei, x_c:x_c+wid], dsize =(c,c), interpolation=cv2.INTER_CUBIC)
                im[-patch_c.shape[0]:,-patch_c.shape[1]:] = patch_c
                rect_params_z = {'xy': (im.shape[1]-patch_c.shape[1]-1, im.shape[0]-patch_c.shape[0]-1), 'width': c, 'height': c, 'linewidth': 2, 'edgecolor': 'red', 'facecolor': 'none'}
                ax.add_patch(plt.Rectangle(**rect_params_z))

                #add a color rectangle
                rect_params_c = {'xy': (x_c, y_c), 'width': wid, 'height': hei, 'linewidth': 2, 'edgecolor': 'red', 'facecolor': 'none'}
                ax.add_patch(plt.Rectangle(**rect_params_c))

                width = 270
                ax.imshow(im)
                rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 0, 'edgecolor': 'black', 'facecolor': 'black'}
                ax.add_patch(plt.Rectangle(**rect_params))
                text_params = {'xy': (5, 5), 'text': r"PSNR$\uparrow$SSIM$\uparrow$BRISQUE$\downarrow$", 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
                ax.annotate(**text_params)
                ax.axis('off')
                ax.set_title("Ground Truth", fontsize=21)
            else:
                im_np = im[0]
                #add a zoom of the image
                patch_c = cv2.resize(im_np[y_c:y_c+hei, x_c:x_c+wid], dsize =(c,c), interpolation=cv2.INTER_CUBIC)
                im_np[-patch_c.shape[0]:,-patch_c.shape[1]:] = patch_c
                rect_params_z = {'xy': (im_np.shape[1]-patch_c.shape[1]-1, im_np.shape[0]-patch_c.shape[0]-1), 'width': c, 'height': c, 'linewidth': 2, 'edgecolor': 'red', 'facecolor': 'none'}
                ax.add_patch(plt.Rectangle(**rect_params_z))

                #add a color rectangle
                rect_params_c = {'xy': (x_c, y_c), 'width': wid, 'height': hei, 'linewidth': 2, 'edgecolor': 'red', 'facecolor': 'none'}
                ax.add_patch(plt.Rectangle(**rect_params_c))

                width = 170
                ax.imshow(im_np)
                rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 0, 'edgecolor': 'black', 'facecolor': 'black'}
                ax.add_patch(plt.Rectangle(**rect_params))
                text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
                ax.annotate(**text_params)
                ax.axis('off')
                if j==1:
                    ax.set_title("Blur", fontsize=21)
                else:
                    ax.set_title("$m = $ "+str(im_name_list[j-2]), fontsize=21)
   
        fig.savefig(path_figure+'/Annealing_influence_images.png')
        plt.show()



if pars.fig_number == 9:
    # Generate a figure with various lambda_0 and various lambda_end for Average PnP
    path_result = "/beegfs/mrenaud/Result_Average_PnP/deblurring/totem/Average_PnP_k_0/noise_10.0/"

    lamb_end_list = [0.1, 0.5, 1., 1.5, 10.]
    lamb_0_list = [0.01, 0.05, 0.1, 0.15, 1.]

    n = len(lamb_0_list)
    m = len(lamb_end_list)

    #size of the black rectangle
    height = 22
    width = 230

    fig = plt.figure(figsize = (n*5, 8))
    for i in range(n):
        dic_PnP = np.load(path_result + "lamb_0_"+str(lamb_0_list[i])+"/annealing_number_16/dict_0_results.npy", allow_pickle=True).item()

        PnP = (dic_PnP["Deblur"], dic_PnP["PSNR_output"], dic_PnP["SSIM_output"], dic_PnP["LPIPS_output"], dic_PnP["BRISQUE_output"])
        
        im = PnP
        ax = fig.add_subplot(1,n,1+i)
        ax.imshow(im[0])
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
        ax.set_title(r"$\alpha_0 = {}$".format(lamb_0_list[i]), fontsize=21)
            
    fig.savefig(path_figure+'/All_results_lamb_0_Average_PnP.png')
    plt.show()

    fig = plt.figure(figsize = (m*5, 8))
    for i in range(n):
        dic_PnP = np.load(path_result + "lamb_end_"+str(lamb_end_list[i])+"/annealing_number_16/dict_0_results.npy", allow_pickle=True).item()

        PnP = (dic_PnP["Deblur"], dic_PnP["PSNR_output"], dic_PnP["SSIM_output"], dic_PnP["LPIPS_output"], dic_PnP["BRISQUE_output"])
        
        im = PnP
        ax = fig.add_subplot(1,m,1+i)
        ax.imshow(im[0])
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
        ax.set_title(r"$\alpha_{m-1} = "+"{}$".format(lamb_end_list[i]), fontsize=21)
            
    fig.savefig(path_figure+'/All_results_lamb_end_Average_PnP.png')
    plt.show()


if pars.fig_number == 10:
    # Generate a figure with various std_0 and various std_end for Average PnP
    path_result = "/beegfs/mrenaud/Result_Average_PnP/deblurring/totem/Average_PnP_k_0/noise_10.0/"

    std_end_list = [0.00392, 0.0117, 0.0196, 0.0392, 0.0706,]
    std_end_name = [1, 3, 5, 10, 18]
    std_0_list = [0.0196, 0.0392, 0.0706, 0.1176]
    std_0_name = [5, 10, 18, 30]

    n = len(std_0_list)
    m = len(std_end_list)

    #size of the black rectangle
    height = 22
    width = 230

    fig = plt.figure(figsize = (n*5, 8))
    for i in range(n):
        dic_PnP = np.load(path_result + "std_0_"+str(std_0_list[i])+"/annealing_number_16/dict_0_results.npy", allow_pickle=True).item()

        PnP = (dic_PnP["Deblur"], dic_PnP["PSNR_output"], dic_PnP["SSIM_output"], dic_PnP["LPIPS_output"], dic_PnP["BRISQUE_output"])
        
        im = PnP
        ax = fig.add_subplot(1,n,1+i)
        ax.imshow(im[0])
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
        ax.set_title(r"$\sigma_0 = {}$".format(std_0_name[i]), fontsize=21)
            
    fig.savefig(path_figure+'/All_results_std_0_Average_PnP.png')
    plt.show()

    fig = plt.figure(figsize = (m*5, 8))
    for i in range(m):
        dic_PnP = np.load(path_result + "std_end_"+str(std_end_list[i])+"/annealing_number_16/dict_0_results.npy", allow_pickle=True).item()

        PnP = (dic_PnP["Deblur"], dic_PnP["PSNR_output"], dic_PnP["SSIM_output"], dic_PnP["LPIPS_output"], dic_PnP["BRISQUE_output"])
        
        im = PnP
        ax = fig.add_subplot(1,m,1+i)
        ax.imshow(im[0])
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': 15, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
        ax.set_title(r"$\sigma_{m-1} = "+"{}$".format(std_end_name[i]), fontsize=21)
            
    fig.savefig(path_figure+'/All_results_std_end_Average_PnP.png')
    plt.show()



if pars.fig_number == 11:
    # Generate a figure with various initialisation of Average PnP
    path_result = "/beegfs/mrenaud/Result_Average_PnP/deblurring/set3c/Average_PnP_k_0/noise_10.0/"

    init_name = ["oracle", "blur", "random"]
    name_fig_list = ["Oracle $x_0$", "Blur $x_0$", "Random $x_0$"]

    n = len(init_name) + 1
    m = 2

    size_title = 25
    size_label = 19

    #size of the black rectangle
    height = 22
    width = 210

    fig = plt.figure(figsize = (n*5, m*5))
    gs = gridspec.GridSpec(2, n, hspace = 0.15, wspace = 0.05)
    for i, name in enumerate(init_name):
        if i ==2:
            width = 220
        dic_APnP = np.load(path_result + "im_init_"+name+"/annealing_number_16/dict_0_results.npy", allow_pickle=True).item()

        APnP = (dic_APnP["Deblur"], dic_APnP["PSNR_output"], dic_APnP["SSIM_output"], dic_APnP["LPIPS_output"], dic_APnP["BRISQUE_output"])
        Init = dic_APnP["Init"]
        GT = dic_APnP["GT"]
        k = dic_APnP["kernel"]
        Blur = (dic_APnP["Blur"], dic_APnP["PSNR_blur"], dic_APnP["SSIM_blur"], dic_APnP["LPIPS_blur"], dic_APnP["BRISQUE_blur"])

        if i == 0:
            ax = plt.subplot(gs[0, 0])
            ax.imshow(Init)
            ax.axis('off')
            ax.set_title("Ground Truth", fontsize=size_title)

            ax = plt.subplot(gs[1, 0])
            im = Blur
            c = 100
            k_resize = cv2.resize(k, dsize =(c,c), interpolation=cv2.INTER_CUBIC)
            im[0][-k_resize.shape[0]:,:k_resize.shape[1]] = k_resize[:,:,None]*np.ones(3)[None,None,:] / np.max(k_resize)
            ax.imshow(im[0])
            rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
            ax.add_patch(plt.Rectangle(**rect_params))
            text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': size_label, 'va': 'top', 'ha': 'left'}
            ax.annotate(**text_params)
            ax.set_title("Observation", fontsize=size_title)
            ax.axis('off')

        ax = plt.subplot(gs[0, i+1])
        ax.imshow(Init)
        ax.axis('off')
        ax.set_title(name_fig_list[i], fontsize=size_title)

        ax = plt.subplot(gs[1, i+1])
        im = APnP
        ax.imshow(im[0])
        rect_params = {'xy': (0, 0), 'width': width, 'height': height, 'linewidth': 1, 'edgecolor': 'black', 'facecolor': 'black'}
        ax.add_patch(plt.Rectangle(**rect_params))
        text_params = {'xy': (5, 5), 'text': "{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(im[1], im[2], im[3], im[4]), 'color': 'white', 'fontsize': size_label, 'va': 'top', 'ha': 'left'}
        ax.annotate(**text_params)
        ax.axis('off')
            
    fig.savefig(path_figure+'/initialisation_sensitivity.png')
    plt.show()