import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import hdf5storage
import deepinv as dinv
from deepinv.utils.plotting import plot
from deepinv.optim.data_fidelity import L2
import os
from natsort import os_sorted
from deepinv.utils.demo import load_url_image, get_image_url
from utils.utils_restoration import single2uint,crop_center, matlab_style_gauss2D, imread_uint, imsave, psnr, array2tensor, tensor2array
from brisque import BRISQUE
from skimage.metrics import structural_similarity as ssim
from lpips import LPIPS
import sys
import wandb
from argparse import ArgumentParser

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

use_wandb = False

if use_wandb:
    # Define sweep config
    sweep_configuration = {
        "method": "grid",
        "name": "sweep",
        "metric": {"goal": "minimize", "name": "output_lpips"},
        "parameters": {
            "lambda_": {"values" : [.05, .1, .15]},
            "zeta": {"values" : [.7, 0.8, 0.9]},
            "diffusion_steps" : {"values" : [20, 50]},
            "t_start" : {"values" : [100, 50, 200]},
        },
    }

    # # # Initialize sweep by passing in config.
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="DiffPIR_deblur")

def restore():
    parser = ArgumentParser()
    parser.add_argument('--lambda_', type=float)
    parser.add_argument('--zeta', type=float)
    parser.add_argument('--diffusion_steps', type=int)
    parser.add_argument('--t_start', type=int)
    parser.add_argument('--dataset_name', type=str, default = "set4c")
    parser.add_argument('--noise_level_img', type=float, default = 10.)
    parser.add_argument('--kernel_index', type=int, default = 0)
    parser.add_argument('--extract_images', type=bool, default = True)
    parser.add_argument('--gpu_number', type=int, default = 0)
    parser.add_argument('--degradation', type= str, default = "deblurring")
    parser_params = parser.parse_args()

    if use_wandb:
        wandb.init()

    if torch.cuda.is_available():
        device = "cuda:"+str(parser_params.gpu_number)
    else:
        device = "cpu"

    loss_lpips = LPIPS(net='alex', version='0.1').to(device)
    brisque = BRISQUE(url=False)

    dataset_path = "../datasets/"
    dataset_name = parser_params.dataset_name

    input_path = os.path.join(dataset_path,dataset_name)
    input_paths = os_sorted([os.path.join(input_path,p) for p in os.listdir(input_path)])

    # #The model of the authors of DiffPIR
    # model = dinv.models.DiffUNet(large_model=False).to(device)

    # The model of GSPnP
    sys.path.append('../GS_denoising/')
    from lightning_GSDRUNet import GradMatch
    parser2 = ArgumentParser(prog='utils_restoration.py')
    parser2 = GradMatch.add_model_specific_args(parser2)
    parser2 = GradMatch.add_optim_specific_args(parser2)
    hparams = parser2.parse_known_args()[0]
    hparams.act_mode = 'E'
    denoiser_model = GradMatch(hparams)
    checkpoint = torch.load('../GS_denoising/ckpts/GSDRUNet.ckpt', map_location=device)
    denoiser_model.load_state_dict(checkpoint['state_dict'],strict=False)
    denoiser_model.eval()
    for i, v in denoiser_model.named_parameters():
        v.requires_grad = False
    denoiser_model = denoiser_model.to(device)

    def model(x, sigma, weight=1.):
        torch.set_grad_enabled(True)
        Dg, N, g = denoiser_model.calculate_grad(x, sigma)
        torch.set_grad_enabled(False)
        Dg = Dg.detach()
        N = N.detach()
        Dx = x - weight * Dg
        return Dx

    psnr_list, ssim_list, lpips_list, brisque_list = [], [], [], []

    method = "DiffPIR"
    exp_out_path = "../../Result_Average_PnP/"+parser_params.degradation+"/"
    exp_out_path = os.path.join(exp_out_path, dataset_name)
    if not os.path.exists(exp_out_path):
        os.mkdir(exp_out_path)
    exp_out_path = os.path.join(exp_out_path, method)
    if not os.path.exists(exp_out_path):
        os.mkdir(exp_out_path)
    if parser_params.degradation == "deblurring":
        exp_out_path = os.path.join(exp_out_path, "kernel_"+str(parser_params.kernel_index))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
        if parser_params.noise_level_img != None:
            exp_out_path = os.path.join(exp_out_path, "noise_level_"+str(parser_params.noise_level_img))
            if not os.path.exists(exp_out_path):
                os.mkdir(exp_out_path)
    if parser_params.lambda_ != None:
        exp_out_path = os.path.join(exp_out_path, "lambda_"+str(parser_params.lambda_))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
    if parser_params.zeta != None:
        exp_out_path = os.path.join(exp_out_path, "zeta_"+str(parser_params.zeta))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
    if parser_params.diffusion_steps != None:
        exp_out_path = os.path.join(exp_out_path, "diffusion_steps_"+str(parser_params.diffusion_steps))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
    if parser_params.t_start != None:
        exp_out_path = os.path.join(exp_out_path, "t_start"+str(parser_params.t_start))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)

    for j in range(len(input_paths)):

        input_im_uint = imread_uint(input_paths[j])
        x_true = array2tensor(np.float32(input_im_uint / 255.)).to(device)

        x = x_true.clone()
        x = x.to(device)
        
        if parser_params.noise_level_img != None:
            sigma_noise = parser_params.noise_level_img /255.
        else:
            sigma_noise = 10. / 255.0  # noise level

        kernel_tensor = torch.from_numpy(k_list[parser_params.kernel_index]).unsqueeze(0).unsqueeze(0).to(device)

        if parser_params.degradation == "deblurring":
            physics = dinv.physics.BlurFFT(
                img_size=(3, x.shape[-2], x.shape[-1]),
                filter=kernel_tensor,
                device=device,
                noise_model=dinv.physics.GaussianNoise(sigma=sigma_noise),
            )
        if parser_params.degradation == "inpainting":
            physics = dinv.physics.Inpainting(
                tensor_size=(3, x.shape[-2], x.shape[-1]),
                mask=0.5,
                device=device,
            )

        y = physics(x)

        T = 1000  # Number of timesteps used during training

        def get_alphas(beta_start=0.1 / 1000, beta_end=20 / 1000, num_train_timesteps=T):
            betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
            betas = torch.from_numpy(betas).to(device)
            alphas = 1.0 - betas
            alphas_cumprod = np.cumprod(alphas.cpu(), axis=0)  # This is \overline{\alpha}_t
            return torch.tensor(alphas_cumprod)

        alphas = get_alphas()

        sigmas = torch.sqrt(1.0 - alphas) / alphas.sqrt()

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx

        data_fidelity = L2()

        #parameters setting
        if use_wandb:
            zeta = wandb.config.zeta
            diffusion_steps = wandb.config.diffusion_steps
            lambda_ = wandb.config.lambda_
            t_start = wandb.config.t_start

        else:
            if parser_params.zeta == None:
                zeta = 0.8
            else:
                zeta = parser_params.zeta
            if parser_params.diffusion_steps == None:
                diffusion_steps = 20
            else:
                diffusion_steps = parser_params.diffusion_steps
            if parser_params.lambda_ == None:
                lambda_ = .13
            else:
                lambda_ = parser_params.lambda_
            if parser_params.t_start == None:
                t_start = 200
            else:
                t_start = parser_params.t_start            

        rhos = lambda_ * (sigma_noise**2) / (sigmas**2)

        # get timestep sequence
        seq = np.sqrt(np.linspace(0, t_start**2, diffusion_steps))
        seq = [int(s) for s in list(seq)]
        seq[-1] = seq[-1] - 1

        # Initialization
        x = 2 * y - 1
        x = torch.clip(torch.sqrt(alphas[t_start]) * x + torch.sqrt(1 - alphas[t_start]) * torch.randn_like(x), 0, 1)

        with torch.no_grad():
            for i in tqdm(range(len(seq))):
                # Current and next noise levels
                curr_sigma = sigmas[t_start - 1 - seq[i]].cpu().numpy()

                # 1. Denoising step
                # print('sigma :', curr_sigma)
                # print("current psnr : {:.2f}dB".format(psnr(np.clip(tensor2array(x_true), 0, 1), np.clip(tensor2array(x), 0, 1))))
                # plt.imsave(exp_out_path+'img_x_'+str(i)+'input.png', single2uint(np.clip(tensor2array(x), 0, 1)))
                x0 = 2*model((x+1)/2, curr_sigma*1.)-1
                # plt.imsave(exp_out_path+'img_x_'+str(i)+'_den_input.png', single2uint(np.clip(tensor2array(x0), 0, 1)))
                # print("current psnr : {:.2f}dB".format(psnr(np.clip(tensor2array(x_true), 0, 1), np.clip(tensor2array(x0), 0, 1))))

                if not seq[i] == seq[-1]:
                    # 2. Data fidelity step
                    t_i = find_nearest(sigmas.cpu(), curr_sigma)

                    x0 = data_fidelity.prox(x0, y, physics, gamma=1 / (2 * rhos[t_i]))

                    # Normalize data for sampling
                    x0 = 2 * x0 - 1
                    x = 2 * x - 1

                    # 3. Sampling step
                    next_sigma = sigmas[t_start - 1 - seq[i + 1]].cpu().numpy()
                    t_im1 = find_nearest(
                        sigmas, next_sigma
                    )  # time step associated with the next noise level

                    eps = (x - alphas[t_i].sqrt() * x0) / torch.sqrt(
                        1.0 - alphas[t_i]
                    )  # effective noise

                    x = alphas[t_im1].sqrt() * x0 + torch.sqrt(1.0 - alphas[t_im1]) * (
                        np.sqrt(1 - zeta) * eps + np.sqrt(zeta) * torch.randn_like(x)
                    )

                    # Rescale the output in [0, 1]
                    x = (x + 1) / 2
        if parser_params.extract_images:
            # Saving images
            plt.imsave(exp_out_path+'/img'+str(j)+'input.png', single2uint(np.clip(tensor2array(x_true), 0, 1)))
            plt.imsave(exp_out_path+'/img'+str(j)+'blur.png', single2uint(np.clip(tensor2array(y), 0, 1)))
            plt.imsave(exp_out_path+'/img'+str(j)+'deblur.png', single2uint(np.clip(tensor2array(x), 0, 1)))
        
        output_psnr = psnr(tensor2array(x_true), tensor2array(x))
        output_ssim = ssim(tensor2array(x_true), tensor2array(x), data_range = 1, channel_axis = 2)
        output_lpips = loss_lpips.forward(x_true, x).item()
        output_brisque = brisque.score(tensor2array(x))
        
        print('PSNR: {:.2f}dB'.format(output_psnr))
        print('SSIM: {:.2f}'.format(output_ssim))
        print('LPIPS: {:.2f}'.format(output_lpips))
        print('BRISQUE: {:.2f}'.format(output_brisque))
        psnr_list.append(output_psnr)
        ssim_list.append(output_ssim)
        lpips_list.append(output_lpips)
        brisque_list.append(output_brisque)

        dict = {
                'GT' : np.clip(tensor2array(x_true), 0, 1),
                'BRISQUE_GT' : brisque.score(np.clip(tensor2array(x_true), 0, 1)),
                'Blur' : np.clip(tensor2array(y), 0, 1),
                'PSNR_blur' : psnr(np.clip(tensor2array(x_true), 0, 1), np.clip(tensor2array(y), 0, 1)),
                'SSIM_blur' : ssim(np.clip(tensor2array(x_true), 0, 1), np.clip(tensor2array(y), 0, 1), data_range = 1, channel_axis = 2),
                'LPIPS_blur' : loss_lpips.forward(x_true, y).item(),
                'BRISQUE_blur' : brisque.score(np.clip(tensor2array(y), 0, 1)),
                'SSIM_output' : output_ssim,
                'PSNR_output' : output_psnr,
                'LPIPS_output' : output_lpips,
                'BRISQUE_output' : output_brisque,
                'zeta' : zeta,
                'diffusion_steps' : diffusion_steps,
                'lambda_' : lambda_
            }
        np.save(os.path.join(exp_out_path, 'dict_' + str(j) + '_results'), dict)

    print("params : lambda = {}, zeta = {}, diffusion_steps = {}, t_start = {}".format(lambda_, zeta, diffusion_steps, t_start))
    print("PSNR Mean :", np.mean(np.array(psnr_list)))
    print("SSIM Mean :", np.mean(np.array(ssim_list)))
    print("LPIPS Mean :", np.mean(np.array(lpips_list)))
    print("BRISQUE Mean :", np.mean(np.array(brisque_list)))

    if use_wandb:
        wandb.log(
            {
                "lambda_": lambda_,
                "diffusion_steps": diffusion_steps,
                "zeta": zeta,
                "t_start": t_start,
                "output_psnr" : np.mean(np.array(psnr_list)),
                "output_ssim" : np.mean(np.array(ssim_list)),
                "output_lpips" : np.mean(np.array(lpips_list)),
                "output_brisque" : np.mean(np.array(brisque_list)),
            }
            )

# # # Start sweep job.
if use_wandb:
    wandb.agent(sweep_id, function=restore)

if __name__ == '__main__' :
    psnr = restore()