import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
from utils.utils_restoration import rescale, psnr, array2tensor, tensor2array, get_gaussian_noise_parameters, create_out_dir, single2uint,crop_center, matlab_style_gauss2D, imread_uint, imsave
import os
import hdf5storage
import sys
from argparse import ArgumentParser
# from models.model_drunet.network_unet import UNetRes as net
import argparse
import cv2
import imageio
from brisque import BRISQUE
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import estimate_sigma
from lpips import LPIPS
from utils.utils_ula import *
from utils import utils_sr

###
# Parser arguments
###

parser = argparse.ArgumentParser()
parser.add_argument("--n_iter", type=int, default=10000, help='number of iteration in PnP-ULA')
parser.add_argument("--alpha", type=float, default=1., help='regularization parameter alpha in PnP-ULA')
parser.add_argument("--s_den", type=int, default=5, help='denoiser parameter')
parser.add_argument("--img", type=str, default='simpson_nb512.png', help='image to reconstruct')
parser.add_argument("--path_result", type=str, default='Result_ULA', help='path to save the results : it will be save in results/path_result')
parser.add_argument("--model_name", type=str, default = 'celebA(woman faces)', help='name of the model for our models')
parser.add_argument("--gpu_number", type=int, default = 0, help='gpu number use')
parser.add_argument("--Lip", type=bool, default = False, help='True : the network is 1-Lip, False : no constraint')
parser.add_argument("--blur_type", type=str, default = 'uniform', help='uniform : uniform blur, gaussian : gaussian blur')
parser.add_argument("--l", type=int, default = 4, help='(2*l+1)*(2*l+1) is the size of the blur kernel. Need to verify 2l+1 < 128')
parser.add_argument("--si", type=float, default = 1., help='std of the blur kernel in case of gaussian blur')
parser.add_argument("--fast_save", type=bool, default = True, help='Make a fast saving of algorithm output')
parser.add_argument("--Pb", type=str, default = 'deblurring', help="Type of problem, possible : 'deblurring', 'inpainting'")
parser.add_argument("--y_path", type=str, default = None, help="If not None, the path of the observation y, use to compute a forward model mismatch")
parser.add_argument("--seed", type=int, default = 40, help="Seed for reproductivity of the algorithm")
pars = parser.parse_args()

###
# PARAMETERS
###

#images used : 'cancer.png', 'castle.png', 'cat.png', 'cells.png', 'duck.png', 'painting.png', 'woman.jpg'

Pb = pars.Pb
N = 256 #size of the image N*N

# Parameters for PnP-ULA
n_iter = pars.n_iter
n_burn_in = int(n_iter/10)
n_inter = int(n_iter/1000)
n_inter_mmse = np.copy(n_inter)

# Denoiser parameters
s = pars.s_den

# Regularization parameters
alpha = pars.alpha # 1 et 0.3
c_min = 0 #-1
c_max = 1 #2

# Inverse problem parameters
sigma = 10
l = pars.l # size of the blurring kernel

# Parameters for the auto-correlation
q = 1000

# Path to save the results
path_result = '../../Result_SNORE/' + pars.path_result + "/" + pars.Pb +  "/alpha_{}".format(pars.alpha) + "/s_{}".format(pars.s_den) + "/n_iter{}".format(pars.n_iter)
os.makedirs(path_result, exist_ok = True)


###
# IMAGE
###
path_img = '../datasets/CBSD10/'
img = '0000.png'
name_im = img.split('.')[0]
im_total = plt.imread(path_img+img)
im = im_total
# if img == "castle.png":
#     im = im_total[100:356,0:256,:]
# else:
#     im = cv2.resize(im_total, dsize=(256, 256)) #expect to have an image at the of format RGB
n_Cols, n_Rows = im.shape[:2]
#image normalization
im = (im - np.min(im)) / (np.max(im) - np.min(im))

###
# Harware Parameters
###

# GPU device selection
cuda = True
device = "cuda:"+str(pars.gpu_number)
#Metrics
loss_lpips = LPIPS(net='alex', version='0.1').to(device)
brisque = BRISQUE(url=False)
# Type
dtype = torch.float32
tensor = torch.FloatTensor
# Seed
seed = pars.seed

# Prior regularization parameter
alphat = torch.tensor(alpha, dtype = dtype, device = device)
# Normalization of the standard deviation noise distribution
sigma1 = sigma/255.0
sigma2 = sigma1**2
sigma2t = torch.tensor(sigma2, dtype = dtype, device = device)
# Normalization of the denoiser noise level
s1 = s/255.
s2 = (s1)**2
s2t = torch.tensor(s2, dtype = dtype, device = device)
# Parameter strong convexity in the tails
lambd = 0.5/(2/sigma2 + alpha/s2)
lambdt = torch.tensor(lambd, dtype = dtype, device = device)

# Discretization step-size
delta = 1/3/(1/sigma2 + 1/lambd + alpha/s2)
deltat = torch.tensor(delta, dtype = dtype, device = device)

###
# Prior-Grad
###

# #load of the pretrained model Drunet
# n_channels = 3 #RGB images
# model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
# path_model = 'Pretrained_models/'+pars.model_name+'.pth'
# model.load_state_dict(torch.load(path_model), strict=True)
# model = model.to(device)

# noise_map = torch.FloatTensor([s1]).repeat(1, 1, 256, 256).to(device)
# def Ds(x):
#     """
#     Denoiser Drunet of level of noise s1 (std of the noise)
#     """
#     x = torch.cat((x, noise_map), dim=1)
#     x = x.to(device)
#     return model(x)

sys.path.append('../GS_denoising/')
from lightning_GSDRUNet import GradMatch
parser2 = ArgumentParser(prog='utils_restoration.py')
parser2 = GradMatch.add_model_specific_args(parser2)
parser2 = GradMatch.add_optim_specific_args(parser2)
hparams = parser2.parse_known_args()[0]
hparams.act_mode = 'E'
hparams.grayscale = False
denoiser_model = GradMatch(hparams)
checkpoint = torch.load('../GS_denoising/ckpts/GSDRUNet.ckpt', map_location=device)
denoiser_model.load_state_dict(checkpoint['state_dict'],strict=False)
denoiser_model.eval()
for i, v in denoiser_model.named_parameters():
    v.requires_grad = False
denoiser_model = denoiser_model.to(device)

def Ds(x, sigma = s1):
    torch.set_grad_enabled(True)
    Dg, N, g = denoiser_model.calculate_grad(x, sigma)
    torch.set_grad_enabled(False)
    Dg = Dg.detach()
    N = N.detach()
    Dx = x - Dg
    return Dx

prior_grad = lambda x : alphat*(Ds(x) - x)/s2t

if Pb == 'inpainting':
    gen = torch.Generator(device=device)
    gen.manual_seed(0) #for reproductivity
    mask = torch.rand((n_Cols, n_Rows), generator=gen, device = device)
    prop = 0.5
    mask = 1*(mask > prop)
    mask = (torch.ones(3)[None,:,None,None].to(device))*mask[None,None,:,:]
    neg_mask = 1 - mask

    im_t = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().unsqueeze(0).to(device)
    y_t = mask * im_t
    prior_grad = lambda x : neg_mask*alphat*(Ds((neg_mask * x + y_t)) - (neg_mask * x + y_t))/s2t
    data_grad = lambda x : 0

    y = y_t.cpu().detach().numpy()
    y = np.transpose(y[0,:,:,:], (1,2,0))
    plt.imsave(path_result + '/observation.png', y) #save the missing pixel image

    #initialization at the Markov Chain
    init_torch = y_t #(torch.mean(y_t)*(3 * 256 * 256)/torch.sum(mask)) * torch.ones((1,3,256,256)).to(device)

###
# Forward model
###

if Pb == 'deblurring':
    # Definition of the convolution kernel
    # if pars.blur_type == 'uniform':
    #     l_h = 2*l+1
    #     h = np.ones((1, l_h))
    # if pars.blur_type == 'gaussian':
    #     si = pars.si
    #     h = np.array([[np.exp(-i**2/(2*si**2)) for i in range(-l,l+1)]])
    # h = h/np.sum(h)
    # kernel= np.dot(h.T,h)

    k_list = []
    # If no specific kernel is given, load the 8 motion blur kernels
    kernel_path = os.path.join('kernels', 'Levin09.mat')
    kernels = hdf5storage.loadmat(kernel_path)['kernels']
    # Kernels follow the order given in the paper (Table 2). The 8 first kernels are motion blur kernels, the 9th kernel is uniform and the 10th Gaussian.
    for k_index in range(10) :
        if k_index == 8: # Uniform blur
            k = (1/81)*np.ones((9,9))
        elif k_index == 9:  # Gaussian blur
            k = matlab_style_gauss2D(shape=(25,25),sigma=1.6)
        else : # Motion blur
            k = kernels[0, k_index]
        k_list.append(k)
    kernel = k_list[0]


    h_conv = np.flip(kernel) # Definition of Data-grad
    h_conv = np.copy(h_conv) # Useful because pytorch cannot handle negatvie strides
    h_conv = np.ones(3)[:,None,None,None] * np.ones(1)[None,:,None,None] *h_conv[None,None,:,:] #have the right dimension of kernels
    h_ = np.ones(3)[:,None,None,None] * np.ones(1)[None,:,None,None] *kernel[None,None,:,:] #have the right dimension of kernels
    hconv_torch = torch.from_numpy(h_conv).type(tensor).to(device)
    hcorr_torch = torch.from_numpy(h_).type(tensor).to(device)

    #forward model definition
    A = lambda x: torch.nn.functional.conv2d(x, hconv_torch, groups=x.size(1), padding = 'same')
    AT = lambda x: torch.nn.functional.conv2d(x, hcorr_torch, groups=x.size(1), padding = 'same')
    
    im_t = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().unsqueeze(0).to(device)

    if pars.y_path == None:
        #blur the blur image in torch
        gen = torch.Generator(device=device)
        gen.manual_seed(0) #for reproductivity
        y_t = A(im_t) + torch.normal(torch.zeros(*im_t.size()).to(device), std = sigma1*torch.ones(*im_t.size()).to(device),generator=gen)
        y_t = torch.clip(y_t, 0, 1)
        y_t = (y_t - torch.min(y_t)) / (torch.max(y_t) - torch.min(y_t))
    else:
        y = plt.imread(pars.y_path)
        y = y[:,:,:3] #convert RGBA to RGB
        y_t = torch.from_numpy(np.ascontiguousarray(y)).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    # DATA-GRAD FOR THE DEBLURRING
    data_grad = lambda x: -AT(A(x) - y_t)/(sigma2t)

    # kernel_t = torch.tensor(kernel).to(device)

    # def data_grad(x, y = y_t):
    #     """
    #     Calculate the gradient of the data-fidelity term.
    #     :param x: Point where to evaluate F
    #     :param y: Degraded image
    #     """
    #     return utils_sr.grad_solution_L2(x, y, kernel_t, sf = 1)

    #initialization at the Markov Chain
    init_torch = y_t

###
# PnP-ULA algorithm
###

# Name for data storage
name = '{}_sigma{}_s{}'.format(name_im, sigma, s)

#PnP-ULA algorithm
Samples_t, Mmse_t, Mmse2_t = pnpula(init = init_torch, data_grad = data_grad, prior_grad = prior_grad, delta = deltat, lambd = lambdt, seed = seed, device = device, n_iter = n_iter, n_inter = n_inter, n_inter_mmse = n_inter_mmse, path = path_result, name = name)

if pars.fast_save:
    #save the observation
    y = y_t.cpu().detach().numpy()
    y = np.transpose(y[0,:,:,:], (1,2,0))
    psb = PSNR(im, y, data_range = 1)
    ssb = ssim(im, y, data_range = 1, channel_axis = 2)

    Mmse = []
    for m in Mmse_t:
        im_ = m.cpu().detach().numpy()
        im_ = np.transpose(im_[:,:,:], (1,2,0))
        Mmse.append(im_)
    xmmse = np.mean(Mmse, axis = 0)
    pmmse = PSNR(im, xmmse, data_range = 1)
    smmse = ssim(im, xmmse, data_range = 1, channel_axis = 2)
    brisquemmse = brisque.score(xmmse)
    xmmse_t = torch.from_numpy(np.ascontiguousarray(xmmse)).permute(2, 0, 1).float().unsqueeze(0).to(device)
    lpipsmmse = loss_lpips.forward(im_t, xmmse_t).item()

    xmmse = (xmmse - np.min(xmmse)) / (np.max(xmmse) - np.min(xmmse))
    # Saving of the MMSE of the sample
    plt.imsave(path_result + '/mmse_' + name + '_psnr{:.2f}_ssim{:.2f}.png'.format(pmmse, smmse), xmmse)
    # Saving of the MMSE compare to the original and observation
    fig = plt.figure(figsize = (10, 10))
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(xmmse)
    ax1.axis('off')
    ax1.set_title("MMSE (PSNR={:.2f}/SSIM={:.2f})".format(pmmse, smmse))
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(im)
    ax2.axis('off')
    ax2.set_title("GT")
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(y)
    ax3.axis('off')
    ax3.set_title("Obs (PSNR={:.2f}/SSIM={:.2f})".format(psb, ssb))
    fig.savefig(path_result+'/MMSE_and_Originale_and_Observation_'+name_im+'n_iter{}'.format(n_iter)+'.png')
    plt.show()

    dict = {
            'Mmse' : Mmse,
            'image_name' : name_im,
            'observation' : y,
            'PSNR_y' : psb,
            'SIM_y' : ssb,
            'ground_truth' : im,
            'MMSE' : xmmse,
            'PSNR_MMSE' : pmmse,
            'SIM_MMSE' : smmse,
            'BRISQUE_MMSE' : brisquemmse,
            'lpipsmmse' : lpipsmmse,
            'n_iter' : n_iter,
            's' : s,
            'alpha' : alpha,
            'c_min' : c_min,
            'c_max' : c_max,
            'sigma' : sigma,
            'l' : l,
            'lambda' : lambd,
            'delta' : delta,
        }

    np.save(path_result+'/'+ name +'_result.npy', dict)



if not(pars.fast_save):

    #convert object in numpy array for analyse
    Samples, Mmse, Mmse2, Psnr_sample, SIM_sample, LPIPS_sample, BRISQUE_sample = [], [], [], [], [], [], []

    for i, sample in enumerate(Samples_t):
        samp = sample.cpu().detach().numpy()
        samp = np.transpose(samp[:,:,:], (1,2,0))
        Psnr_sample.append(PSNR(im, samp, data_range = 1))
        SIM_sample.append(ssim(im, samp, data_range = 1, channel_axis = 2))
        LPIPS_sample.append(loss_lpips.forward(im_t, sample).item())
        BRISQUE_sample.append(brisque.score(samp))
        Samples.append(samp)

    for m in Mmse_t:
        im_ = m.cpu().detach().numpy()
        im_ = np.transpose(im_[:,:,:], (1,2,0))
        Mmse.append(im_)
    for m in Mmse2_t:
        im_ = m.cpu().detach().numpy()
        im_ = np.transpose(im_[:,:,:], (1,2,0))
        Mmse2.append(im_)

    #save the observation
    y = y_t.cpu().detach().numpy()
    y = np.transpose(y[0,:,:,:], (1,2,0))
    psb = PSNR(im, y, data_range = 1)
    ssb = ssim(im, y, data_range = 1, channel_axis = 2)

    # Compute PSNR and SIM for the online MMSE
    n = len(Mmse)
    PSNR_list, SIM_list, LPIPS_list, BRISQUE_list = [], [], [], []

    mean_list = np.cumsum(Mmse, axis = 0) / np.arange(1,n+1)[:,None,None,None]
    for i in range(1,n):
        mean = mean_list[i]
        PSNR_list.append(PSNR(im, mean, data_range = 1))
        SIM_list.append(ssim(im, mean, data_range = 1, channel_axis = 2))
        BRISQUE_list.append(brisque.score(mean))
        mean_t = torch.from_numpy(np.ascontiguousarray(mean)).permute(2, 0, 1).float().unsqueeze(0).to(device)
        LPIPS_list.append(loss_lpips.forward(im_t, mean_t).item())

    # Computation of the mean and std of the whole Markov chain
    xmmse = np.mean(Mmse, axis = 0)
    pmmse = PSNR(im, xmmse, data_range = 1)
    smmse = ssim(im, xmmse, data_range = 1, channel_axis = 2)
    brisquemmse = brisque.score(xmmse)
    xmmse_t = torch.from_numpy(np.ascontiguousarray(xmmse)).permute(2, 0, 1).float().unsqueeze(0).to(device)
    lpipsmmse = loss_lpips.forward(im_t, xmmse_t).item()

    # Computation of the std of the Markov chain
    xmmse2 = np.mean(Mmse2, axis = 0)
    var = xmmse2 - xmmse**2
    var = var*(var>=0) + 0*(var<0)
    std = np.sum(np.sqrt(var), axis = -1)
    diff = np.sum(np.abs(im-xmmse), axis= -1)

    #save the result of the experiment
    dict = {
            'Samples' : Samples,
            'Mmse' : Mmse,
            'Mmse2' : Mmse2,
            'PSNR_sample' : Psnr_sample,
            'SIM_sample' : SIM_sample,
            'LPIPS_sample' : LPIPS_sample,
            'BRISQUE_sample' : BRISQUE_sample,
            'PSNR_mmse' : PSNR_list,
            'SIM_list' : SIM_list,
            'LPIPS_list' : LPIPS_list,
            'BRISQUE_list' : BRISQUE_list,
            'image_name' : name_im,
            'observation' : y,
            'PSNR_y' : psb,
            'SIM_y' : ssb,
            'ground_truth' : im,
            'MMSE' : xmmse,
            'PSNR_MMSE' : pmmse,
            'SIM_MMSE' : smmse,
            'BRISQUE_MMSE' : brisquemmse,
            'lpipsmmse' : lpipsmmse,
            'std' : std,
            'diff' : diff,
            'n_iter' : n_iter,
            's' : s,
            'alpha' : alpha,
            'c_min' : c_min,
            'c_max' : c_max,
            'sigma' : sigma,
            'l' : l,
            'lambda' : lambd,
            'delta' : delta,
        }

    np.save(path_result+'/'+ name +'_result.npy', dict)

    #save the observation
    plt.imsave(path_result + '/observation'+name_im+'.png', y)

    # #creation of a video of the samples
    # writer = imageio.get_writer(os.path.join(path_result,"samples_video"+name+".mp4"), fps=100)
    # for im_ in Samples:
    #     im_uint8 = np.clip(im_ * 255, 0, 255).astype(np.uint8)
    #     writer.append_data(im_uint8)
    # writer.close()

    # PSNR plots
    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(Psnr_sample, "+")
    ax.set_title("PSNR between samples and GT")
    fig.savefig(path_result +"/PSNR_between_samples_and_GT_"+name_im+"n_iter{}".format(n_iter)+".png")
    plt.show()

    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(PSNR_list, "+")
    ax.set_title("PSNR between online MMSE and GT")
    fig.savefig(path_result +"/PSNR_between_online_MMSE_and_GT_"+name_im+"n_iter{}".format(n_iter)+".png")
    plt.show()

    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(SIM_sample, "+")
    ax.set_title("SIM between samples and GT")
    fig.savefig(path_result +"/SIM_between_samples_and_GT_"+name_im+"n_iter{}".format(n_iter)+".png")
    plt.show()

    fig, ax = plt.subplots(figsize = (10,10))
    ax.plot(SIM_list, "+")
    ax.set_title("SIM between online MMSE and GT")
    fig.savefig(path_result +"/SIM_between_online_MMSE_and_GT_"+name_im+"n_iter{}".format(n_iter)+".png")
    plt.show()

    xmmse = (xmmse - np.min(xmmse)) / (np.max(xmmse) - np.min(xmmse))

    std = (std - np.min(std)) / (np.max(std) - np.min(std))
    diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))

    # Saving of the standard deviation and the difference between MMSE and Ground-Truth (GT)
    fig = plt.figure(figsize = (10, 5))
    ax1 = fig.add_subplot(1,2,1)
    ax1.imshow(std, cmap = 'gray')
    ax1.axis('off')
    ax1.set_title("Std of the Markov Chain")
    ax2 = fig.add_subplot(1,2,2)
    ax2.imshow(diff, cmap = 'gray')
    ax2.axis('off')
    ax2.set_title("Diff MMSE-GT")
    fig.savefig(path_result+'/Std_of_the_Markov_Chain_'+name_im+'n_iter{}'.format(n_iter)+'.png')
    plt.show()

    # Saving of the Fourier transforme of the standard deviation, to detect possible artecfact of sampling
    plt.imsave(path_result+"/Fourier_transform_std_MC_"+name_im+'n_iter{}'.format(n_iter)+".png",np.fft.fftshift(np.log(np.abs(np.fft.fft2(std))+1e-10)))