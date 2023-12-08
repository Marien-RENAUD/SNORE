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

loss_lpips = LPIPS(net='alex', version='0.1')
brisque = BRISQUE(url=False)

class PnP_restoration():

    def __init__(self, hparams):

        self.hparams = hparams
        self.device = torch.device('cuda:'+str(self.hparams.gpu_number) if torch.cuda.is_available() else 'cpu')
        self.initialize_cuda_denoiser()

    def initialize_cuda_denoiser(self):
        '''
        Initialize the denoiser model with the given pretrained ckpt
        '''
        sys.path.append('../GS_denoising/')
        from lightning_GSDRUNet import GradMatch
        parser2 = ArgumentParser(prog='utils_restoration.py')
        parser2 = GradMatch.add_model_specific_args(parser2)
        parser2 = GradMatch.add_optim_specific_args(parser2)
        hparams = parser2.parse_known_args()[0]
        hparams.act_mode = self.hparams.act_mode_denoiser
        self.denoiser_model = GradMatch(hparams)
        checkpoint = torch.load(self.hparams.pretrained_checkpoint, map_location=self.device)
        self.denoiser_model.load_state_dict(checkpoint['state_dict'],strict=False)
        self.denoiser_model.eval()
        for i, v in self.denoiser_model.named_parameters():
            v.requires_grad = False
        self.denoiser_model = self.denoiser_model.to(self.device)

    def denoise(self, x, sigma, weight=1.):
        if self.hparams.rescale_for_denoising:
            mintmp = x.min()
            maxtmp = x.max()
            x = (x - mintmp) / (maxtmp - mintmp)
        elif self.hparams.clip_for_denoising:
            x = torch.clamp(x,0,1)
        torch.set_grad_enabled(True)
        Dg, N, g = self.denoiser_model.calculate_grad(x, sigma)
        torch.set_grad_enabled(False)
        Dg = Dg.detach()
        N = N.detach()
        if self.hparams.rescale_for_denoising:
            N = N * (maxtmp - mintmp) + mintmp
        Dx = x - weight * Dg
        return Dx, g, Dg


    def initialize_prox(self, img, degradation):
        '''
        calculus for future prox computatations
        :param img: degraded image
        :param degradation: 2D blur kernel for deblurring and SR, mask for inpainting
        '''
        if self.hparams.degradation_mode == 'deblurring' or self.hparams.degradation_mode == 'SR':
            k = degradation
            self.k_tensor = torch.tensor(k).to(self.device)
            self.FB, self.FBC, self.F2B, self.FBFy = utils_sr.pre_calculate_prox(img, self.k_tensor, self.sf)
        elif self.hparams.degradation_mode == 'inpainting':
            self.M = array2tensor(degradation).to(self.device)

    def data_fidelity_prox_step(self, x, y, stepsize):
        '''
        Calculation of the proximal step on the data-fidelity term f
        '''
        if self.hparams.noise_model == 'gaussian':
            if self.hparams.degradation_mode == 'deblurring' or self.hparams.degradation_mode == 'SR':
                px = utils_sr.prox_solution_L2(x, self.FB, self.FBC, self.F2B, self.FBFy, stepsize, self.sf)
            elif self.hparams.degradation_mode == 'inpainting':
                if self.hparams.noise_level_img > 1e-2:
                    px = (stepsize*self.M*y + x)/(stepsize*self.M+1)
                else :
                    px = self.M*y + (1-self.M)*x
            else:
                ValueError('Degradation not treated')
        else :  
            ValueError('noise model not treated')
        return px

    def data_fidelity_grad(self, x, y):
        if self.hparams.noise_model == 'gaussian':
            return utils_sr.grad_solution_L2(x, y, self.k_tensor, self.sf)
        else:
            raise ValueError('noise model not implemented')   

    def data_fidelity_grad_step(self, x, y, stepsize):
        '''
        Calculation of the gradient step on the data-fidelity term f
        '''
        if self.hparams.noise_model == 'gaussian':
            grad = utils_sr.grad_solution_L2(x, y, self.k_tensor, self.sf)
        else:
            raise ValueError('noise model not implemented')
        return x - stepsize*grad, grad
        
    def A(self,y):
        '''
        Calculation A*x with A the linear degradation operator 
        '''
        if self.hparams.degradation_mode == 'deblurring':
            y = utils_sr.G(y, self.k_tensor, sf=1)
        elif self.hparams.degradation_mode == 'SR':
            y = utils_sr.G(y, self.k_tensor, sf=self.sf)
        elif self.hparams.degradation_mode == 'inpainting':
            y = self.M * y
        else:
            raise ValueError('degradation not implemented')
        return y  

    def At(self,y):
        '''
        Calculation A*x with A the linear degradation operator 
        '''
        if self.hparams.degradation_mode == 'deblurring':
            y = utils_sr.Gt(y, self.k_tensor, sf=1)
        elif self.hparams.degradation_mode == 'SR':
            y = utils_sr.Gt(y, self.k_tensor, sf=self.sf)
        elif self.hparams.degradation_mode == 'inpainting':
            y = self.M * y
        else:
            raise ValueError('degradation not implemented')
        return y  


    def calulate_data_term(self,y,img):
        '''
        Calculation of the data term value f(y)
        :param y: Point where to evaluate F
        :param img: Degraded image
        :return: f(y)
        '''
        deg_y = self.A(y)
        if self.hparams.noise_model == 'gaussian':
            f = 0.5 * torch.norm(img - deg_y, p=2) ** 2
        elif self.hparams.noise_model == 'poisson':
            f = (img*torch.log(img/deg_y + 1e-15) + deg_y - img).sum()
        return f

    def calculate_regul(self,x, g=None):
        '''
        Calculation of the regularization phi_sigma(y)
        :param x: Point where to evaluate
        :param g: Precomputed regularization function value at x
        :return: regul(x)
        '''
        if g is None:
            _,g,_ = self.denoise(x, self.sigma_denoiser)
        return g


    def calculate_F(self,x, img, g = None):
        '''
        Calculation of the objective function value f(x) + lamb*g(x)
        :param x: Point where to evaluate F
        :param img: Degraded image
        :param g: Precomputed regularization function value at x
        :return: F(x)
        '''
        regul = self.calculate_regul(x, g=g)
        if self.hparams.no_data_term:
            F = regul
            f = torch.zeros_like(F)
        else:
            f = self.calulate_data_term(x,img)
            F = f + self.lamb * regul
        return f.item(), F.item()

    def restore(self, img, init_im, clean_img, degradation,extract_results=False, sf=1):
        '''
        Compute GS-PnP restoration algorithm
        :param img: Degraded image
        :param init_im: Initialization of the algorithm
        :param clean_img: ground-truth clean image
        :param degradation: 2D blur kernel for deblurring and SR, mask for inpainting
        :param extract_results: Extract information for subsequent image or curve saving
        :param sf: Super-resolution factor
        '''

        self.sf = sf
        if self.hparams.opt_alg == "Average_PnP" or self.hparams.opt_alg == "Average_PnP_Prox":
            self.hparams.use_backtracking = False
            self.hparams.early_stopping = False

        if extract_results:
            y_list, z_list, x_list, Dg_list, psnr_tab, ssim_tab, brisque_tab, lpips_tab, g_list, f_list, Df_list, F_list, Psi_list = [], [], [], [], [], [], [], [], [], [], [], [], []

        # initalize parameters
        if (self.hparams.opt_alg == "PnP_Prox" or self.hparams.opt_alg == "PnP_GD" or self.hparams.opt_alg == "Data_GD"):
            if self.hparams.stepsize is None:
                self.stepsize = 1 / self.lamb
            else:
                self.stepsize = self.hparams.stepsize

        i = 0 # iteration counter

        img_tensor = array2tensor(img).to(self.device) # for GPU computations (if GPU available)
        self.initialize_prox(img_tensor, degradation) # prox calculus that can be done outside of the loop

        # Initialization of the algorithm
        x0 = array2tensor(init_im).to(self.device)

        if self.hparams.use_linear_init:
            x0 = self.At(x0)
        if self.hparams.use_hard_constraint:
            x0 = torch.clamp(x0, 0, 1)
        x0 = self.data_fidelity_prox_step(x0, img_tensor, self.stepsize)
        x = x0

        if extract_results:  # extract np images and PSNR values
            out_x = tensor2array(x0.cpu())
            current_x_psnr = psnr(clean_img, out_x)
            if self.hparams.print_each_step:
                print('current x PSNR : ', current_x_psnr)
            psnr_tab.append(current_x_psnr)
            x_list.append(out_x)

        x = x0

        diff_F = 1
        F = float('inf')
        self.backtracking_check = True
        
        if self.hparams.opt_alg == "PnP_SGD":
            lamb = 1.
            std = 20. / 255.
            L_tot = (lamb/std**2 + 1 / self.hparams.noise_level_img**2)
            delta_stable = 2 / L_tot
            delta_0 = delta_stable / 6

            i = 0
            psnr_down = -float("inf")
            psnr_up = current_x_psnr
            while i < 200: #and abs(psnr_up - psnr_down) > 0.1 * delta_0:
                # print(i)
                # print(abs(psnr_up - psnr_down))
                x_old = x
                if i % 50 == 0:
                    imsave('deblurring/test_x_' + str(i) + '.png', single2uint(tensor2array(x_old.cpu())))
                _,g,Dg = self.denoise(x_old, std)
                x_denoised = x_old - Dg
                if i % 50 == 0:
                    imsave('deblurring/test_xdenoised_' + str(i) + '.png', single2uint(tensor2array(x_denoised.cpu())))
                noise = torch.normal(torch.zeros(*x_old.size()).to(self.device), std = torch.ones(*x_old.size()).to(self.device))
                x = x_old - delta_0 * lamb * Dg / std**2 - delta_0 * self.data_fidelity_grad(x_old, img_tensor) + delta_0 * noise
                z = x

                f, F = self.calculate_F(x_old, img_tensor, g=g)
                out_z = tensor2array(z.cpu())
                out_x = tensor2array(x.cpu())
                current_z_psnr = psnr(clean_img, out_z)
                current_x_psnr = psnr(clean_img, out_x)
                if self.hparams.print_each_step:
                    print('iteration : ', i)
                    print('current z PSNR : ', current_z_psnr)
                    print('current x PSNR : ', current_x_psnr)
                x_list.append(out_x)
                z_list.append(out_z)
                g_list.append(g.cpu().item())
                Dg_list.append(torch.norm(Dg).cpu().item())
                psnr_tab.append(current_z_psnr)
                current_z_ssim = ssim(clean_img, out_z, data_range = 1, channel_axis = 2)
                ssim_tab.append(current_z_ssim)
                F_list.append(F)
                f_list.append(f)
                psnr_down = psnr_up
                psnr_up = current_z_psnr
                i += 1

            print("number of burn-in iteration : ", i)
            i = 1
            while i < 401:
                delta_i = delta_0/(i**0.8)
                x_old = x
                _,g,Dg = self.denoise(x_old, std)
                z = x_old - delta_i * lamb * Dg / std**2 - delta_i * self.data_fidelity_grad(x_old, img_tensor)
                noise = torch.normal(torch.zeros(*x_old.size()).to(self.device), std = torch.ones(*x_old.size()).to(self.device))
                x = z + delta_i * noise
                f, F = self.calculate_F(x_old, img_tensor, g=g)
                if extract_results:
                    out_z = tensor2array(z.cpu())
                    out_x = tensor2array(x.cpu())
                    current_z_psnr = psnr(clean_img, out_z)
                    current_x_psnr = psnr(clean_img, out_x)
                    if self.hparams.print_each_step:
                        print('iteration : ', i)
                        print('current z PSNR : ', current_z_psnr)
                        print('current x PSNR : ', current_x_psnr)
                    x_list.append(out_x)
                    z_list.append(out_z)
                    g_list.append(g.cpu().item())
                    Dg_list.append(torch.norm(Dg).cpu().item())
                    psnr_tab.append(current_z_psnr)
                    current_z_ssim = ssim(clean_img, out_z, data_range = 1, channel_axis = 2)
                    ssim_tab.append(current_z_ssim)
                    F_list.append(F)
                    f_list.append(f)
                i += 1
            y = x

        if self.hparams.opt_alg != "PnP_SGD":
            for i in tqdm(range(self.maxitr)):

                F_old = F
                x_old = x
                
                # # The 50 first steps are special for inpainting
                # if self.hparams.inpainting_init and i < self.hparams.n_init:
                #     self.sigma_denoiser = 50
                #     use_backtracking = False
                #     early_stopping = False
                # else :
                #     self.sigma_denoiser = self.hparams.sigma_denoiser
                #     use_backtracking = self.hparams.use_backtracking
                #     early_stopping = self.hparams.early_stopping

                if self.hparams.opt_alg == "Data_GD":
                    z = x_old
                    # Data-fidelity step
                    x = self.data_fidelity_prox_step(z, img_tensor, self.stepsize)
                    y = z # output image is the output of the denoising step
                    if self.hparams.use_hard_constraint:
                        x = torch.clamp(x,0,1)
                    # Calculate Objective
                    g=torch.tensor(0).float()
                    Dg=torch.tensor(0).float()
                    f, F = self.calculate_F(x, img_tensor, g=g)

                if self.hparams.opt_alg == "PnP_Prox" or self.hparams.opt_alg == "PnP_GD":
                    # Gradient of the regularization term
                    _,g,Dg = self.denoise(x_old, self.sigma_denoiser)
                    # Gradient step
                    z = x_old - self.stepsize * self.lamb * Dg
                    # Data-fidelity step
                    if self.hparams.opt_alg == "PnP_Prox":
                        x = self.data_fidelity_prox_step(z, img_tensor, self.stepsize)
                    if self.hparams.opt_alg == "PnP_GD":
                        x = z - self.stepsize * self.data_fidelity_grad(x_old, img_tensor)
                    y = z # output image is the output of the denoising step
                    if self.hparams.use_hard_constraint:
                        x = torch.clamp(x,0,1)
                    # Calculate Objective
                    f, F = self.calculate_F(x, img_tensor, g=g)

                    
                if self.hparams.opt_alg == "Average_PnP" or self.hparams.opt_alg == "Average_PnP_Prox":
                    x_old = x
                    num_itr_each_ann = (self.maxitr - 300) // self.hparams.annealing_number
                    if i % num_itr_each_ann == 0 and i < self.maxitr - 300:
                        self.std =  self.std_0 * (1 - i / (self.maxitr - 300)) + self.std_end * (i / (self.maxitr - 300))
                        self.lamb = self.lamb_0 * (1 - i / (self.maxitr - 300)) + self.lamb_end * (i / (self.maxitr - 300))
                    if i >= self.maxitr - 300:
                        self.std = self.std_end
                        self.lamb = self.lamb_end
                    # if i >= 3*self.maxitr//4:
                    #     self.stepsize = self.hparams.stepsize / i**0.8
                    # Regularization term
                    g_mean = torch.tensor([0]).to(self.device).float()
                    Dg_mean = torch.zeros(*x_old.size()).to(self.device)
                    for _ in range(self.hparams.num_noise):
                        noise = torch.normal(torch.zeros(*x_old.size()).to(self.device), std = self.std*torch.ones(*x_old.size()).to(self.device))
                        x_old_noise = x_old + noise
                        _,g,Dg = self.denoise(x_old_noise, self.std)
                        g_mean += g
                        Dg_mean += Dg
                    g, Dg = g_mean/self.hparams.num_noise, Dg_mean/self.hparams.num_noise
                    # Total-Gradient step
                    z = x_old - self.stepsize * self.lamb * Dg 
                    if self.hparams.opt_alg == "Average_PnP":
                        x = z - self.stepsize * self.data_fidelity_grad(x_old, img_tensor)
                    if self.hparams.opt_alg == "Average_PnP_Prox":
                        x = self.data_fidelity_prox_step(z, img_tensor, self.stepsize)
                    # print("reg = ",torch.sum(torch.abs(self.lamb * Dg)))
                    # print("data = ",torch.sum(torch.abs(self.data_fidelity_grad(x_old, img_tensor))))
                    # Hard constraint
                    if self.hparams.use_hard_constraint:
                        x = torch.clamp(x,0,1)
                    # Calculate Objective
                    f, F = self.calculate_F(x, img_tensor, g=g)

                    y = x # output image is the output of the denoising step
                    z = x # To be modified, for no errors in the followinf code       

                # Backtracking
                if self.hparams.use_backtracking :
                    diff_x = (torch.norm(x - x_old, p=2) ** 2)
                    diff_F = F_old - F
                    if diff_F < (self.hparams.gamma_backtracking / self.stepsize) * diff_x :
                        self.stepsize = self.hparams.eta_backtracking * self.stepsize
                        self.backtracking_check = False
                        print('backtracking : stepsize =', self.stepsize, 'diff_F=', diff_F)
                    else :
                        self.backtracking_check = True
                    # if (abs(self.stepsize) < 1e-7):
                    #     print(f'Convergence reached at iteration {i}')
                    #     break

                if self.backtracking_check : # if the backtracking condition is satisfied
                    # Logging
                    if extract_results:
                        out_z = tensor2array(z.cpu())
                        out_x = tensor2array(x.cpu())
                        current_z_psnr = psnr(clean_img, out_z)
                        current_x_psnr = psnr(clean_img, out_x)
                        if self.hparams.print_each_step:
                            print('iteration : ', i)
                            print('current z PSNR : ', current_z_psnr)
                            print('current x PSNR : ', current_x_psnr)
                            print('current F : ', F)
                        x_list.append(out_x)
                        z_list.append(out_z)
                        g_list.append(g.cpu().item())
                        Dg_list.append(torch.norm(Dg).cpu().item())
                        psnr_tab.append(current_z_psnr)
                        current_z_ssim = ssim(clean_img, out_z, data_range = 1, channel_axis = 2)
                        ssim_tab.append(current_z_ssim)
                        brisque_tab.append(brisque.score(out_z))
                        clean_img_tensor, out_z_tensor = array2tensor(clean_img).float(), array2tensor(out_z).float()
                        if self.hparams.lpips:
                            current_z_lpips = loss_lpips.forward(clean_img_tensor, out_z_tensor).item()
                            lpips_tab.append(current_z_lpips)
                        F_list.append(F)
                        f_list.append(f)

                    # check decrease of data_fidelity 
                    if self.hparams.early_stopping : 
                        if self.hparams.crit_conv == 'cost':
                            if (abs(diff_F)/abs(F) < self.hparams.thres_conv):
                                print(f'Convergence reached at iteration {i}')
                                break
                        elif self.hparams.crit_conv == 'residual':
                            diff_x = torch.norm(x - x_old, p=2)
                            if diff_x/torch.norm(x) < self.hparams.thres_conv:
                                print(f'Convergence reached at iteration {i}')
                                break

                    # i += 1 # next iteration

                else : # if the backtracking condition is not satisfied
                    x = x_old
                    F = F_old

        if self.hparams.opt_alg == "PnP_GD":#a last denoising
            Dy,_,_ = self.denoise(y, self.sigma_denoiser)
            y = Dy

        output_img = tensor2array(y.cpu())
        output_psnr = psnr(clean_img, output_img)
        output_ssim = ssim(clean_img, output_img, data_range = 1, channel_axis = 2)
        output_brisque = brisque.score(output_img)
        clean_img_tensor, output_img_tensor = array2tensor(clean_img).float(), array2tensor(output_img).float()
        output_lpips = loss_lpips.forward(clean_img_tensor, output_img_tensor).item()

        if extract_results:
            return output_img, tensor2array(x0.cpu()), output_psnr, output_ssim, output_lpips, output_brisque, i, x_list, z_list, np.array(Dg_list), np.array(psnr_tab), np.array(ssim_tab), np.array(brisque_tab), np.array(lpips_tab), np.array(g_list), np.array(F_list), np.array(f_list)
        else:
            return output_img, tensor2array(x0.cpu()), output_psnr, output_ssim, output_lpips, output_brisque, i

    def initialize_curves(self):

        self.conv = []
        self.conv_F = []
        self.PSNR = []
        self.SSIM = []
        self.BRISQUE = []
        self.LPIPS = []
        self.g = []
        self.Dg = []
        self.F = []
        self.f = []
        self.lip_algo = []
        self.lip_D = []
        self.lip_Dg = []

    def update_curves(self, x_list, psnr_tab, ssim_tab, brisque_tab, lpips_tab, Dg_list, g_list, F_list, f_list):
        self.F.append(F_list)
        self.f.append(f_list)
        self.g.append(g_list)
        self.Dg.append(Dg_list)
        self.PSNR.append(psnr_tab)
        self.SSIM.append(ssim_tab)
        self.BRISQUE.append(brisque_tab)
        self.LPIPS.append(lpips_tab)
        self.conv.append(np.array([(np.linalg.norm(x_list[k + 1] - x_list[k]) ** 2) for k in range(len(x_list) - 1)]) / np.sum(np.abs(x_list[0]) ** 2))
        self.lip_algo.append(np.sqrt(np.array([np.sum(np.abs(x_list[k + 1] - x_list[k]) ** 2) for k in range(1, len(x_list) - 1)]) / np.array([np.sum(np.abs(x_list[k] - x_list[k - 1]) ** 2) for k in range(1, len(x_list[:-1]))])))

    def save_curves(self, save_path):

        import matplotlib
        matplotlib.rcParams.update({'font.size': 17})
        matplotlib.rcParams['lines.linewidth'] = 2
        matplotlib.style.use('seaborn-darkgrid')
        use_tex = matplotlib.checkdep_usetex(True)
        if use_tex:
            plt.rcParams['text.usetex'] = True

        plt.figure(0)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.g)):
            plt.plot(self.g[i], '-o')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'g.png'),bbox_inches="tight")

        plt.figure(1)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.g)):
            plt.plot(self.g[i][self.maxitr//2+1:], '-o')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'g_end.png'), bbox_inches="tight")

        plt.figure(2)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.PSNR)):
            plt.plot(self.PSNR[i], '-o')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'PSNR.png'),bbox_inches="tight")

        plt.figure(3)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.SSIM)):
            plt.plot(self.SSIM[i], '-o')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'SSIM.png'),bbox_inches="tight")

        plt.figure(4)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.LPIPS)):
            plt.plot(self.LPIPS[i], '-o')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'LPIPS.png'),bbox_inches="tight")

        plt.figure(5)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.F)):
            plt.plot(self.F[i], '-o')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'F.png'), bbox_inches="tight")

        plt.figure(6)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        for i in range(len(self.F)):
            plt.plot(self.F[i][self.maxitr//2+1:], '-o')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'F_end.png'), bbox_inches="tight")

        plt.figure(7)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.F)):
            plt.plot(self.f[i], '-o')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'f.png'), bbox_inches="tight")

        plt.figure(8)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.f)):
            plt.plot(self.f[i][self.maxitr//2+1:], '-o')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'f_end.png'), bbox_inches="tight")

        # conv_DPIR = np.load('conv_DPIR2.npy')
        conv_rate = self.conv[0][0]*np.array([(1/k) for k in range(1,len(self.conv[0]))])
        plt.figure(9)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.conv)):
            plt.plot(self.conv[i],'-o')
            # plt.plot(conv_DPIR[:self.hparams.maxitr], marker=marker_list[-1], markevery=10, label='DPIR')
        plt.plot(conv_rate, '--', color='red', label=r'$\mathcal{O}(\frac{1}{K})$')
        plt.semilogy()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'conv_log.png'), bbox_inches="tight")

        plt.figure(10)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.lip_algo)):
            plt.plot(self.lip_algo[i],'-o')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'lip_algo.png'))

        plt.figure(11)
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(len(self.BRISQUE)):
            plt.plot(self.BRISQUE[i], '-o')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig(os.path.join(save_path, 'BRISQUE.png'),bbox_inches="tight")


    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dataset_path', type=str, default='../datasets')
        parser.add_argument('--gpu_number', type=int, default=0)
        parser.add_argument('--pretrained_checkpoint', type=str,default='../GS_denoising/ckpts/GSDRUNet.ckpt')
        parser.add_argument('--noise_model', type=str,  default='gaussian')
        parser.add_argument('--dataset_name', type=str, default='set3c')
        parser.add_argument('--noise_level_img', type=float, required=True)
        parser.add_argument('--maxitr', type=int)
        parser.add_argument('--stepsize', type=float)
        parser.add_argument('--lamb', type=float)
        parser.add_argument('--std_0', type=float)
        parser.add_argument('--std_end', type=float)
        parser.add_argument('--lamb_0', type=float)
        parser.add_argument('--lamb_end', type=float)
        parser.add_argument('--num_noise', type=int, default=1)
        parser.add_argument('--annealing_number', type=int, default=16)
        parser.add_argument('--sigma_denoiser', type=float)
        parser.add_argument('--lpips', dest='lpips', action='store_true')
        parser.set_defaults(lpips=False)
        parser.add_argument('--n_images', type=int, default=68)
        parser.add_argument('--crit_conv', type=str, default='cost')
        parser.add_argument('--thres_conv', type=float, default=1e-5)
        parser.add_argument('--no_backtracking', dest='use_backtracking', action='store_false')
        parser.set_defaults(use_backtracking=True)
        parser.add_argument('--eta_backtracking', type=float, default=0.9)
        parser.add_argument('--gamma_backtracking', type=float, default=0.1)
        parser.add_argument('--inpainting_init', dest='inpainting_init', action='store_true')
        parser.set_defaults(inpainting_init=False)
        parser.add_argument('--extract_curves', dest='extract_curves', action='store_true')
        parser.set_defaults(extract_curves=False)
        parser.add_argument('--extract_images', dest='extract_images', action='store_true')
        parser.set_defaults(extract_images=False)
        parser.add_argument('--save_video', dest='save_video', action='store_true')
        parser.set_defaults(save_video=False)
        parser.add_argument('--print_each_step', dest='print_each_step', action='store_true')
        parser.set_defaults(print_each_step=False)
        parser.add_argument('--no_data_term', dest='no_data_term', action='store_true')
        parser.set_defaults(no_data_term=False)
        parser.add_argument('--use_hard_constraint', dest='use_hard_constraint', action='store_true')
        parser.set_defaults(use_hard_constraint=False)
        parser.add_argument('--rescale_for_denoising', dest='rescale_for_denoising', action='store_true')
        parser.set_defaults(rescale_for_denoising=False)
        parser.add_argument('--clip_for_denoising', dest='clip_for_denoising', action='store_true')
        parser.set_defaults(clip_for_denoising=False)
        parser.add_argument('--use_wandb', dest='use_wandb', action='store_true')
        parser.set_defaults(use_wandb=False)
        parser.add_argument('--use_linear_init', dest='use_linear_init', action='store_true')
        parser.set_defaults(use_linear_init=False)
        parser.add_argument('--grayscale', dest='grayscale', action='store_true')
        parser.set_defaults(grayscale=False)
        parser.add_argument('--no_early_stopping', dest='early_stopping', action='store_false')
        parser.set_defaults(early_stopping=True)
        parser.add_argument('--weight_Dg', type=float, default=1.)
        parser.add_argument('--n_init', type=int, default=10)
        parser.add_argument('--act_mode_denoiser', type=str, default='E')
        parser.add_argument('--opt_alg', dest='opt_alg', choices=['Average_PnP', 'Data_GD', 'Average_PnP_Prox', 'PnP_Prox', 'PnP_GD', 'PnP_AGD', 'PnP_SGD'], help='Specify optimization algorithm')
        return parser