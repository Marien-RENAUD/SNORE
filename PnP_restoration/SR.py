import os
import numpy as np
import hdf5storage
from scipy import ndimage
from argparse import ArgumentParser
from utils.utils_restoration import psnr, modcrop, rescale, array2tensor, tensor2array, get_gaussian_noise_parameters, create_out_dir, single2uint,crop_center, matlab_style_gauss2D, imread_uint, imsave
from skimage.metrics import structural_similarity as ssim
from natsort import os_sorted
from GS_PnP_restoration import PnP_restoration
import wandb
import cv2
from utils.utils_sr import numpy_degradation, shift_pixel
from lpips import LPIPS
from brisque import BRISQUE

loss_lpips = LPIPS(net='alex', version='0.1')
brisque = BRISQUE(url=False)

def SR():

    parser = ArgumentParser()
    parser.add_argument('--sf', nargs='+', type=int)
    parser.add_argument('--kernel_path', type=str)
    parser.add_argument('--kernel_indexes', nargs='+', type=int)
    parser.add_argument('--image_path', type=str)
    parser = PnP_restoration.add_specific_args(parser)
    hparams = parser.parse_args()

    # SR specific hyperparameters
    hparams.degradation_mode = 'SR'
    hparams.classical_degradation = True

    # PnP_restoration class
    PnP_module = PnP_restoration(hparams)

    # Set input image paths
    if hparams.image_path is not None : # if a specific image path is given
        input_paths = [hparams.image_path]
        hparams.dataset_name = os.path.splitext(os.path.split(hparams.image_path)[-1])[0]
    else : # if not given, we aply on the whole dataset name given in argument 
        input_path = os.path.join(hparams.dataset_path,hparams.dataset_name)
        input_paths = os_sorted([os.path.join(input_path,p) for p in os.listdir(input_path)])

    psnr_list = []
    ssim_list = []
    F_list = []

    if hparams.kernel_path is not None : # if a specific kernel saved in hparams.kernel_path as np array is given 
        k_list = [np.load(hparams.kernel_path)]
        k_index_list = [0]
    else : 
        k_list = []
        # If no specific kernel is given, load the 8 blur kernels
        kernel_path = os.path.join('kernels','kernels_12.mat')
        kernels = hdf5storage.loadmat(kernel_path)['kernels']
        # Kernels follow the order given in the paper (Table 2). The 8 first kernels are motion blur kernels, the 9th kernel is uniform and the 10th Gaussian.
        for k_index in range(8) :
            k = kernels[0, k_index]
            k_list.append(k)
        if hparams.kernel_indexes is not None : 
            k_index_list = hparams.kernel_indexes
        else :
            k_index_list = range(len(k_list))

    if hparams.sf is not None : # if SR scales are given 
        sf_list = hparams.sf
    else :
        sf_list = [2]


    #create the folder to save experimental results
    exp_out_path = "../../Result_SNORE"
    if not os.path.exists(exp_out_path):
        os.mkdir(exp_out_path)
    exp_out_path = os.path.join(exp_out_path, hparams.degradation_mode)
    if not os.path.exists(exp_out_path):
        os.mkdir(exp_out_path)
    exp_out_path = os.path.join(exp_out_path, hparams.dataset_name)
    if not os.path.exists(exp_out_path):
        os.mkdir(exp_out_path)
    exp_out_path = os.path.join(exp_out_path, PnP_module.hparams.opt_alg)
    if not os.path.exists(exp_out_path):
        os.mkdir(exp_out_path)
    exp_out_path = os.path.join(exp_out_path, "sf_"+str(PnP_module.hparams.sf))
    if not os.path.exists(exp_out_path):
        os.mkdir(exp_out_path)
    exp_out_path = os.path.join(exp_out_path, "noise_"+str(PnP_module.hparams.noise_level_img))
    if not os.path.exists(exp_out_path):
        os.mkdir(exp_out_path)
    if PnP_module.hparams.maxitr != None:
        exp_out_path = os.path.join(exp_out_path, "maxitr_"+str(PnP_module.maxitr))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
    if PnP_module.hparams.seed != None:
        exp_out_path = os.path.join(exp_out_path, "seed_"+str(PnP_module.hparams.seed))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
    if PnP_module.hparams.stepsize != None:
        exp_out_path = os.path.join(exp_out_path, "stepsize_"+str(PnP_module.stepsize))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
    if PnP_module.hparams.lamb_0 != None:
        exp_out_path = os.path.join(exp_out_path, "lamb_0_"+str(PnP_module.hparams.lamb_0))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
    if PnP_module.hparams.lamb_end != None:
        exp_out_path = os.path.join(exp_out_path, "lamb_end_"+str(PnP_module.hparams.lamb_end))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
    if PnP_module.hparams.std_0 != None:
        exp_out_path = os.path.join(exp_out_path, "std_0_"+str(PnP_module.hparams.std_0))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
    if PnP_module.hparams.std_end != None:
        exp_out_path = os.path.join(exp_out_path, "std_end_"+str(PnP_module.hparams.std_end))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
    if PnP_module.hparams.lamb != None:
        exp_out_path = os.path.join(exp_out_path, "lamb_"+str(PnP_module.hparams.lamb))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
    if PnP_module.hparams.sigma_denoiser != None:
        exp_out_path = os.path.join(exp_out_path, "sigma_denoiser_"+str(PnP_module.hparams.sigma_denoiser))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
    if PnP_module.hparams.no_data_term == True:
        exp_out_path = os.path.join(exp_out_path, "no_data_term")
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
    if PnP_module.hparams.annealing_number != None:
        exp_out_path = os.path.join(exp_out_path, "annealing_number_"+str(PnP_module.hparams.annealing_number))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
    if PnP_module.hparams.num_noise != 1:
        exp_out_path = os.path.join(exp_out_path, "num_noise_"+str(PnP_module.hparams.num_noise))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)

    if hparams.use_wandb :
        wandb.init(project=hparams.degradation_mode)        
    data = []

    for k_index in k_index_list : # For each kernel

        psnr_k_list = []
        ssim_k_list = []
        n_it_list = []

        k = k_list[k_index]

        for sf in sf_list :

            if hparams.extract_curves:
                PnP_module.initialize_curves()

            if PnP_module.hparams.opt_alg == 'RED_Prox':
                PnP_module.lamb, PnP_module.sigma_denoiser, PnP_module.maxitr, PnP_module.thres_conv = get_gaussian_noise_parameters(
                                        hparams.noise_level_img, hparams, k_index=k_index, degradation_mode='SR')
                PnP_module.std = PnP_module.sigma_denoiser
                PnP_module.lamb_0 = PnP_module.lamb_end = PnP_module.lamb
                PnP_module.std_0 = PnP_module.std_end = PnP_module.std
            if PnP_module.hparams.opt_alg == 'SNORE_Prox' or PnP_module.hparams.opt_alg == 'SNORE':
                PnP_module.lamb, PnP_module.sigma_denoiser, PnP_module.maxitr, PnP_module.thres_conv = get_gaussian_noise_parameters(
                                        hparams.noise_level_img, hparams, k_index=k_index, degradation_mode='SR')
                PnP_module.std_0 = 4. * PnP_module.hparams.noise_level_img / 255.
                PnP_module.std = PnP_module.std_end = 2. * PnP_module.hparams.noise_level_img / 255.
                PnP_module.lamb_end = 0.3
                PnP_module.lamb_0 = 0.02
                PnP_module.stepsize = 1.
                PnP_module.hparams.last_itr = 100

            print('GS-DRUNET super-resolution with image sigma:{:.3f}, model sigma:{:.3f}, lamb:{:.3f} \n'.format(PnP_module.hparams.noise_level_img, PnP_module.sigma_denoiser, PnP_module.lamb))

            for i in range(min(len(input_paths),hparams.n_images)) : # For each image

                print('SR of image {}, sf={}, kernel index {}'.format(i, sf, k_index))
                
                np.random.seed(seed=0)
                # load image
                input_im_uint = imread_uint(input_paths[i])
                input_im_uint = input_im_uint[:sf*int(input_im_uint.shape[0]/sf), :sf*int(input_im_uint.shape[1]/sf), :]
                input_im = np.float32(input_im_uint / 255.)
                #to have a size multiple of the sf factor

                if hparams.grayscale : 
                    input_im = cv2.cvtColor(input_im, cv2.COLOR_BGR2GRAY)
                    input_im = np.expand_dims(input_im, axis = 2)
                # Degrade image
                blur_im = modcrop(input_im, sf)
                blur_im = numpy_degradation(input_im, k, sf)
                noise = np.random.normal(0, hparams.noise_level_img/255., blur_im.shape)
                blur_im += noise
                # Initialize the algorithm
                init_im = cv2.resize(blur_im, (int(blur_im.shape[1] * sf), int(blur_im.shape[0] * sf)),interpolation=cv2.INTER_CUBIC)
                init_im = shift_pixel(init_im, sf)

                # PnP restoration
                if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
                    deblur_im, init_im, output_psnr, output_ssim, output_lpips, output_brisque, output_den_img, output_den_psnr, output_den_ssim, output_den_brisque, output_den_img_tensor, output_den_lpips, n_it, x_list, z_list, Dg_list, psnr_tab, ssim_tab, brisque_tab, lpips_tab, g_list, F_list, f_list, lamb_tab, std_tab = PnP_module.restore(blur_im.copy(),init_im.copy(),input_im.copy(),k, extract_results=True, sf=sf)        
                else:
                    deblur_im, init_im, output_psnr, output_ssim, output_lpips, output_brisque, _, _, _, _, _, _, _ = PnP_module.restore(blur_im.copy(),init_im.copy(),input_im.copy(),k, extract_results=True, sf=sf)

                print('PSNR: {:.2f}dB'.format(output_psnr))
                print('SSIM: {:.2f}'.format(output_ssim))
                print('LPIPS: {:.2f}'.format(output_lpips))
                print('BRISQUE: {:.2f}'.format(output_brisque))
                print(f'N iterations: {n_it}')
                
                psnr_k_list.append(output_psnr)
                ssim_k_list.append(output_ssim)
                psnr_list.append(output_psnr)
                ssim_list.append(output_ssim)
                n_it_list.append(n_it)

                if hparams.extract_curves:
                    # Create curves
                    PnP_module.update_curves(x_list, psnr_tab, ssim_tab, brisque_tab, lpips_tab, Dg_list, g_list, F_list, f_list, lamb_tab, std_tab)

                if hparams.extract_images:
                    # Save images
                    save_im_path = os.path.join(exp_out_path, 'images')
                    if not os.path.exists(save_im_path):
                        os.mkdir(save_im_path)
                    imsave(os.path.join(save_im_path, 'img_'+str(i)+'_input.png'), input_im_uint)
                    imsave(os.path.join(save_im_path, 'img_' + str(i) + '_deblur.png'), single2uint(np.clip(deblur_im, 0, 1)))
                    imsave(os.path.join(save_im_path, 'img_'+str(i)+'_blur.png'), single2uint(np.clip(blur_im, 0, 1)))
                    imsave(os.path.join(save_im_path, 'img_' + str(i) + '_init.png'), single2uint(np.clip(init_im, 0, 1)))
                    print('output image saved at ', os.path.join(save_im_path, 'img_' + str(i) + '_deblur.png'))

                    #save the result of the experiment
                    input_im_tensor, blur_im_tensor = array2tensor(input_im).float(), array2tensor(blur_im).float()
                    dict = {
                            'GT' : input_im,
                            'BRISQUE_GT' : brisque.score(input_im),
                            'Deblur' : deblur_im,
                            'Blur' : blur_im,
                            'Init' : init_im,
                            'SSIM_output' : output_ssim,
                            'PSNR_output' : output_psnr,
                            'LPIPS_output' : output_lpips,
                            'BRISQUE_output' : output_brisque,
                            'kernel' : k,
                            'lamb' : PnP_module.lamb,
                            'lamb_0' : PnP_module.lamb_0,
                            'lamb_end' : PnP_module.lamb_end,
                            'maxitr' : PnP_module.maxitr,
                            'std_0' : PnP_module.std_0,
                            'std_end' : PnP_module.std_end,
                            'stepsize' : PnP_module.stepsize,
                            'opt_alg': PnP_module.hparams.opt_alg,
                            'psnr_tab' : psnr_tab,
                            'ssim_tab' : ssim_tab,
                            'brisque_tab' : brisque_tab,
                            'lpips_tab' : lpips_tab,
                            'Dg_list' : Dg_list,
                            'g_list' : g_list,
                            'F_list' : F_list,
                            'f_list' : f_list,
                            'lamb_tab' : lamb_tab,
                            'std_tab' : std_tab,
                            'output_den_img' : output_den_img, 
                            'output_den_psnr' : output_den_psnr, 
                            'output_den_ssim' : output_den_ssim, 
                            'output_den_lpips' : output_den_lpips,
                            'output_den_brisque' : output_den_brisque, 
                        }
                    np.save(os.path.join(exp_out_path, 'dict_' + str(i) + '_results'), dict)

            if hparams.extract_curves:
                # Save curves
                save_curves_path = os.path.join(exp_out_path,'curves')
                if not os.path.exists(save_curves_path):
                    os.mkdir(save_curves_path)
                PnP_module.save_curves(save_curves_path)
                print('output curves saved at ', save_curves_path)

        avg_k_psnr = np.mean(np.array(psnr_k_list))
        avg_k_ssim = np.mean(np.array(ssim_k_list))
        print('avg RGB psnr on kernel {} : {:.2f}dB'.format(k_index, avg_k_psnr))
        print('avg RGB ssim on kernel {} : {:.2f}'.format(k_index, avg_k_ssim))
        data.append([k_index, sf, np.mean(np.mean(n_it_list))])

    data = np.array(data)
    if hparams.use_wandb :
        table = wandb.Table(data=data, columns=['k', 'psnr', 'n_it'])
        for i, metric in enumerate(['psnr', 'n_it']):
            wandb.log({
                f'{metric}_plot': wandb.plot.scatter(
                    table, 'k', metric,
                    title=f'{metric} vs. k'),
                f'average_{metric}': np.mean(data[:,i+1])
            },
            step = 0)

    return np.mean(np.array(psnr_list))


if __name__ == '__main__':
    SR()