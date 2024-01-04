import os
import numpy as np
from collections import OrderedDict
from argparse import ArgumentParser
from GS_PnP_restoration import PnP_restoration
from utils.utils_restoration import single2uint,crop_center, matlab_style_gauss2D, imread_uint, imsave
from natsort import os_sorted
import wandb

# Define sweep config
sweep_configuration = {
    "method": "grid",
    "name": "sweep",
    "metric": {"goal": "minimize", "name": "output_lpips"},
    "parameters": {
        "lamb": {"values" : [0.1, 0.05, 0.15]},
        "sigma_denoiser": {"values" : [10. / 255., 5. / 255., 15. / 255.]},
        "maxitr" : {"values" : [500]},
    },
}

# # # Initialize sweep by passing in config.
sweep_id = wandb.sweep(sweep=sweep_configuration, project="Average_PnP")

def inpaint():

    if hparams.use_wandb:
        wandb.init()    

    parser = ArgumentParser()
    parser.add_argument('--prop_mask', type=float, default=0.5)
    parser.add_argument('--image_path', type=str)
    parser = PnP_restoration.add_specific_args(parser)
    hparams = parser.parse_args()

    # Inpainting specific hyperparameters
    
    hparams.degradation_mode = 'inpainting'
    hparams.noise_level_img = 0
    hparams.n_init = 10
    hparams.use_backtracking = False
    hparams.inpainting_init = True

    # PnP_restoration class
    PnP_module = PnP_restoration(hparams)

    PnP_module.lamb, PnP_module.lamb_0, PnP_module.lamb_end, PnP_module.maxitr, PnP_module.std_0, PnP_module.std_end, PnP_module.stepsize = PnP_module.hparams.lamb, PnP_module.hparams.lamb_0, PnP_module.hparams.lamb_end, PnP_module.hparams.maxitr, PnP_module.hparams.std_0, PnP_module.hparams.std_end, PnP_module.hparams.stepsize
    PnP_module.hparams.std = hparams.sigma_denoiser
    PnP_module.hparams.early_stopping = False

    if PnP_module.std_end == None:
        if PnP_module.hparams.opt_alg == 'Average_PnP_Prox':
            PnP_module.std_end = 5. / 255.
        if PnP_module.hparams.opt_alg == 'PnP_Prox':
            PnP_module.sigma_denoiser = 10. / 255.
    if PnP_module.maxitr == None:
        PnP_module.maxitr = 500
    if PnP_module.std_0 == None and PnP_module.hparams.opt_alg == 'Average_PnP_Prox':
        PnP_module.std_0 = 50. /255.
    if PnP_module.stepsize == None and PnP_module.hparams.opt_alg == 'Average_PnP_Prox':
        PnP_module.stepsize = 1.
    if PnP_module.lamb == None and PnP_module.hparams.opt_alg == 'PnP_Prox':
        PnP_module.lamb = 0.3
    if PnP_module.lamb_0 == None and PnP_module.hparams.opt_alg == 'Average_PnP_Prox':
        PnP_module.lamb_0 = 0.1
    if PnP_module.lamb_end == None and PnP_module.hparams.opt_alg == 'Average_PnP_Prox':
        PnP_module.lamb_end = 0.1

    if hparams.use_wandb:
        PnP_module.lamb = wandb.config.lamb
        PnP_module.sigma_denoiser = wandb.config.sigma_denoiser
        PnP_module.maxitr = wandb.config.maxitr

     # Set input image paths
    if hparams.image_path is not None : # if a specific image path is given
        input_paths = [hparams.image_path]
        hparams.dataset_name = os.path.splitext(os.path.split(hparams.image_path)[-1])[0]
    else : # if not given, we aply on the whole dataset name given in argument 
        input_path = os.path.join(hparams.dataset_path,hparams.dataset_name)
        input_paths = os_sorted([os.path.join(input_path,p) for p in os.listdir(input_path)])

    #create the folder to save experimental results
    exp_out_path = "../../Result_Average_PnP"
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
    exp_out_path = os.path.join(exp_out_path, "noise_"+str(PnP_module.hparams.noise_level_img))
    if not os.path.exists(exp_out_path):
        os.mkdir(exp_out_path)
    if PnP_module.hparams.maxitr != None:
        exp_out_path = os.path.join(exp_out_path, "maxitr_"+str(PnP_module.maxitr))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
    exp_out_path = os.path.join(exp_out_path, "mask_prop_"+str(hparams.prop_mask))
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

    test_results = OrderedDict()
    test_results['psnr'] = []

    if hparams.extract_curves:
        PnP_module.initialize_curves()

    psnr_list, ssim_list, lpips_list, brisque_list = [], [], [], []
    psnrY_list = []
    F_list = []

    for i in range(min(len(input_paths), hparams.n_images)): # For each image

        print('__ image__', i)

        # load image
        input_im_uint = imread_uint(input_paths[i])
        # if hparams.patch_size < min(input_im_uint.shape[0], input_im_uint.shape[1]):
        #     input_im_uint = crop_center(input_im_uint, hparams.patch_size, hparams.patch_size)
        input_im = np.float32(input_im_uint / 255.)
        # Degrade image
        mask = np.random.binomial(n=1, p=hparams.prop_mask, size=(input_im.shape[0],input_im.shape[1]))
        mask = np.expand_dims(mask,axis=2)
        mask_im = input_im*mask + (0.5)*(1-mask)

        # no noise is added
        # np.random.seed(seed=0)
        # mask_im += np.random.normal(0, hparams.noise_level_img/255., mask_im.shape)

        # PnP restoration
        if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
            inpainted_im, _, output_psnr, output_ssim, output_lpips, output_brisque, output_den_img, output_den_psnr, output_den_ssim, output_den_brisque, output_den_img_tensor, output_den_lpips,_, x_list, z_list, Dg_list, psnr_tab, ssim_tab, brisque_tab, lpips_tab, g_list, F_list, f_list, lamb_tab, std_tab = PnP_module.restore(mask_im, mask_im, input_im, mask, extract_results=True)        
        else:
            inpainted_im, _, output_psnr, output_ssim, output_lpips, output_brisque, _, _, _, _, _, _, _ = PnP_module.restore(mask_im, mask_im, input_im, mask)

        print('PSNR: {:.2f}dB'.format(output_psnr))
        print('SSIM: {:.2f}'.format(output_ssim))
        print('LPIPS: {:.2f}'.format(output_lpips))
        print('BRISQUE: {:.2f}'.format(output_brisque))
        psnr_list.append(output_psnr)
        ssim_list.append(output_ssim)
        lpips_list.append(output_lpips)
        brisque_list.append(output_brisque)
        # psnrY_list.append(output_psnrY)

        if hparams.extract_curves:
            # Create curves
            PnP_module.update_curves(x_list, psnr_tab, ssim_tab, brisque_tab, lpips_tab, Dg_list, g_list, F_list, f_list, lamb_tab, std_tab)

        if hparams.extract_images:
            # Save images
            save_im_path = os.path.join(exp_out_path, 'images')
            if not os.path.exists(save_im_path):
                os.mkdir(save_im_path)

            imsave(os.path.join(save_im_path, 'img_'+str(i)+'_input.png'), input_im_uint)
            imsave(os.path.join(save_im_path, 'img_' + str(i) + "_inpainted.png"), single2uint(np.clip(inpainted_im, 0, 1)))
            imsave(os.path.join(save_im_path, 'img_'+str(i)+'_masked.png'), single2uint(np.clip(mask_im*mask, 0, 1)))
            imsave(os.path.join(save_im_path, 'img_' + str(i) + '_init.png'), single2uint(np.clip(mask_im, 0, 1)))

            print('output images saved at ', save_im_path)

        
    if hparams.extract_curves:
        # Save curves
        save_curves_path = os.path.join(exp_out_path, 'curves')
        if not os.path.exists(save_curves_path):
            os.mkdir(save_curves_path)
        PnP_module.save_curves(save_curves_path)
        print('output curves saved at ', save_curves_path)

    if hparams.use_wandb:
        wandb.log(
            {
                "std_0": PnP_module.std_0,
                "std_end": PnP_module.std_end,
                "lamb_0": PnP_module.lamb_0,
                "lamb_end": PnP_module.lamb_end,
                "stepsize": PnP_module.stepsize,
                "maxitr": PnP_module.maxitr,
                "output_psnr" : np.mean(np.array(psnr_list)),
                "output_ssim" : np.mean(np.array(ssim_list)),
                "output_lpips" : np.mean(np.array(lpips_list)),
                "output_brisque" : np.mean(np.array(brisque_list)),
            }
            )

    avg_k_psnr = np.mean(np.array(psnr_list))
    print('avg RGB psnr : {:.2f}dB'.format(avg_k_psnr))
    # avg_k_psnrY = np.mean(np.array(psnrY_list))
    # print('avg Y psnr : {:.2f}dB'.format(avg_k_psnrY))

# # # Start sweep job.
wandb.agent(sweep_id, function=inpaint)

if __name__ == '__main__' :
    psnr = inpaint()