import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
import os
import utils
# from models.model_drunet.denoiser import Denoiser
import argparse
import imageio
import cv2

def load_image_gray(path_img, img):
    """
    Load the image img from path_image and crop it to have a gray scale image 256*256

    Parameters
    ----------
    path_img : str
        path where img is saved
    img : str
        name of the image

    Returns
    -------
    im : ndarray, shape (256, 256)
        The load image
    """
    im_total = plt.imread(path_img+img)
    if img == "duck.png":
        im = np.mean(im_total, axis = 2)
        im = im[400:656,550:806]
    if img == "painting.png" or img == 'cancer.png' or img == 'cells.png':
        im = np.mean(im_total, axis = 2)
        im = cv2.resize(im, dsize=(256, 256))
    if img == "castle.png":
        im = np.mean(im_total, axis = 2)
        im = im[100:356,0:256]
    if img == "simpson_nb512.png" or img == "goldhill.png":
        im = im_total[100:356,0:256]
    if img in set(["cameraman.png", '01.png', '02.png', '03.png', '04.png', '05.png', '06.png', '07.png']):
        im = im_total
    if img == '09.png':
        im = im_total[:256,256:]
    if img == '10.png':
        im = im_total[100:356,256:]
    if img == '11.png':
        im = im_total[:256,:256]
    if img == '12.png':
        im = im_total[100:356,100:356]
    return im

def torch_denoiser(x,model):
    """
    pytorch_denoiser for a denoiser train to predict the noise
    Inputs:
        xtilde      noisy tensor
        model       pytorch denoising model
   
    Output:
        x           denoised tensor
    """

    # denoise
    with torch.no_grad():
        #xtorch = xtilde.unsqueeze(0).unsqueeze(0)
        r = model(x)
        #r = np.reshape(r, -1)
        x_ = x - r
        out = torch.squeeze(x_)
    return out

def pnpula(init, data_grad, prior_grad, delta, lambd, n_iter = 5000, n_inter = 1000, n_inter_mmse = 1000, seed = None, device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu"), c_min = -1, c_max = 2, path = None, name = None, save_online = False):
    """
    PnP-ULA sampling algorithm 

    Inputs:
        init        Initialization of the Markov chain (torch tensor)
        prior_grad  Gradient of the log-prior (already multiplied by the regularization parameter)
        data_grad   Gradient of the likelihood
        delta       Discretization step-size (torch tensor)
        lambd       Moreau-Yoshida regularization parameter (torch tensor)
        n_iter      Number of ULA iterations
        n_inter     Number of iterations before saving of a sample
        n_inter_mmse Number of iterations for a mean computation
        device      cuda device used to store samples
        seed        int, seed used
        path        str : where to store the data
        name        name of the stored dictionnary
        c_min       To ensure strong convexity
        c_max       To ensure strong convexity
    Outputs:
        Xlist       Samples stored every n_inter iterations
        Xlist_mmse  Mmse computed over n_inter_mmse iterations
        Xlist_mmse2 Average X**2 computed over n_inter_mmse iterations
    """
    # Type
    dtype = torch.float32
    tensor = torch.FloatTensor
    # Shape of the image
    im_shape = init.shape
    # Markov chain init
    X = torch.zeros(im_shape, dtype = dtype, device = device)
    X = init.clone().detach()
    # To compute the empirical average over n_inter_mmse
    xmmse = torch.zeros(im_shape, dtype = dtype, device = device)
    # To compute the empirical variance over n_inter_mmse
    xmmse2 = torch.zeros(im_shape, dtype = dtype, device = device)
    # 
    One = torch.ones(xmmse2.shape)
    One = One.to(device)

    # 
    brw = torch.sqrt(2*delta)
    brw = brw.to(device)
 
    if path == None:
        path = str()   
    if seed == None:
        seed = 1
        torch.manual.seed(seed)
        np.random.seed(seed)  
    if name == None:
        name = str()
    if n_inter_mmse == None:
        n_inter_mmse = np.copy(n_inter)

    #To store results
    Xlist = []
    Xlist_mmse = []
    Xlist_mmse2 = []
    iter_mmse = 0

    # Frequency at which we save samples
    K = int(n_iter/10)
    
    with torch.no_grad():
        for i in tqdm(range(n_iter)):
            Z = torch.randn(im_shape, dtype = dtype, device = device)
            # grad G : Tweedie's formula
            grad_log_prior = prior_grad(X)
            # grad F : gaussian Data fit
            grad_log_data = data_grad(X) # T comment if we want to sample from the prior
            # Projection
            out = torch.where(X>c_min, X, c_min*One)
            proj = torch.where(out<c_max, out, c_max*One)
            # grad log-posterior
            gradPi = grad_log_prior - (X-proj)/lambd  + grad_log_data
            # Langevin update
            X = X + delta*gradPi + brw*Z

            # To save samples of the Markov chain after the burn-in every n_inter iterations.
            if i%n_inter == 0:
                X_ = torch.squeeze(X)
                # Sample Storage
                Xlist.append(X_)

            # Computation online of E[X] and E[X**2]
            if iter_mmse <= n_inter_mmse-1:
                xmmse = iter_mmse/(iter_mmse + 1)*xmmse + 1/(iter_mmse + 1)*X
                xmmse2 = iter_mmse/(iter_mmse + 1)*xmmse2 + 1/(iter_mmse + 1)*X**2
                iter_mmse += 1
            else:
                xmmse = iter_mmse/(iter_mmse + 1)*xmmse + 1/(iter_mmse + 1)*X
                xmmse2 = iter_mmse/(iter_mmse + 1)*xmmse2 + 1/(iter_mmse + 1)*X**2
                z = torch.squeeze(xmmse)
                z2 = torch.squeeze(xmmse2)
                Xlist_mmse.append(z)
                Xlist_mmse2.append(z2)
                del xmmse
                del xmmse2
                xmmse = torch.zeros(im_shape, dtype = dtype, device = device)
                xmmse2 = torch.zeros(im_shape, dtype = dtype, device = device)
                iter_mmse = 0
            # Saving the data on the disk during the process
            if i%K == 0 and save_online:
                #save the result of the experiment
                dict = {
                        'Samples' : Xlist,
                        'Mmse' : Xlist_mmse,
                        'Mmse2' : Xlist_mmse2,
                        'n_iter' : n_iter,
                        'c_min' : c_min,
                        'c_max' : c_max,
                        'lambda' : lambd,
                        'delta' : delta,
                    }
                torch.save(dict, path+'/'+ name +'_sampling.pth')

    return Xlist, Xlist_mmse, Xlist_mmse2



