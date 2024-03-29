import os
import numpy as np
from collections import OrderedDict
from argparse import ArgumentParser
from GS_PnP_restoration import PnP_restoration
from utils.utils_restoration import rescale, single2uint,crop_center, matlab_style_gauss2D, imread_uint, imsave, psnr, array2tensor, tensor2array
from natsort import os_sorted
import wandb
import imageio
from brisque import BRISQUE
from skimage.metrics import structural_similarity as ssim
from skimage.restoration import estimate_sigma
from lpips import LPIPS

loss_lpips = LPIPS(net='alex', version='0.1')
brisque = BRISQUE(url=False)

# # Define sweep config
# sweep_configuration = {
#     "method": "grid",
#     "name": "sweep",
#     "metric": {"goal": "minimize", "name": "output_lpips"},
#     "parameters": {
#         "lamb_end": {"values" : [0.4, 0.5, 0.6, 0.3, 0.7]},
#         "std_end": {"values" : [5.]},
#         "maxitr" : {"values" : [500]},
#     },
# }

# # # # Initialize sweep by passing in config.
# sweep_id = wandb.sweep(sweep=sweep_configuration, project="SNORE_inpainting")

def inpaint():
    parser = ArgumentParser()
    parser.add_argument('--prop_mask', type=float, default=0.5)
    parser.add_argument('--image_path', type=str)
    parser = PnP_restoration.add_specific_args(parser)
    hparams = parser.parse_args()

    if hparams.use_wandb:
        wandb.init()  

    # Inpainting specific hyperparameters
    hparams.degradation_mode = 'inpainting'
    hparams.noise_level_img = 0
    hparams.n_init = 10
    hparams.use_backtracking = False
    hparams.inpainting_init = True

    # PnP_restoration class
    PnP_module = PnP_restoration(hparams)

    PnP_module.lamb, PnP_module.lamb_0, PnP_module.lamb_end, PnP_module.maxitr, PnP_module.std_0, PnP_module.std_end, PnP_module.stepsize, PnP_module.beta = PnP_module.hparams.lamb, PnP_module.hparams.lamb_0, PnP_module.hparams.lamb_end, PnP_module.hparams.maxitr, PnP_module.hparams.std_0, PnP_module.hparams.std_end, PnP_module.hparams.stepsize, PnP_module.hparams.beta
    PnP_module.sigma_denoiser = PnP_module.std_end

    if PnP_module.std_end == None:
        if PnP_module.hparams.opt_alg == 'SNORE_Prox' or PnP_module.hparams.opt_alg == 'SNORE':
            PnP_module.std_end = 5. / 255.
        if PnP_module.hparams.opt_alg == 'RED_Prox' or PnP_module.hparams.opt_alg == 'RED':
            PnP_module.sigma_denoiser = 10. / 255.
    if PnP_module.maxitr == None:
        PnP_module.maxitr = 500
    if PnP_module.std_0 == None and (PnP_module.hparams.opt_alg == 'SNORE_Prox' or PnP_module.hparams.opt_alg == 'SNORE'):
        PnP_module.std_0 = 50. /255.
    if PnP_module.stepsize == None and PnP_module.hparams.opt_alg == 'SNORE_Prox':
        PnP_module.stepsize = 1.
    if PnP_module.stepsize == None and PnP_module.hparams.opt_alg == 'SNORE':
        PnP_module.stepsize = .5
    if PnP_module.lamb == None and PnP_module.hparams.opt_alg == 'RED_Prox' or PnP_module.hparams.opt_alg == 'RED':
        PnP_module.lamb = 0.15
    if PnP_module.hparams.opt_alg == 'RED':
        PnP_module.hparams.n_init = 100
        PnP_module.hparams.stepsize = 0.5
    if PnP_module.lamb_0 == None and (PnP_module.hparams.opt_alg == 'SNORE_Prox' or PnP_module.hparams.opt_alg == 'SNORE'):
        PnP_module.lamb_0 = 0.15
    if PnP_module.lamb_end == None and PnP_module.hparams.opt_alg == 'SNORE_Prox':
        PnP_module.lamb_end = 0.15
    if PnP_module.lamb_end == None and  PnP_module.hparams.opt_alg == 'SNORE':
        PnP_module.lamb_end = 0.4
    
    if PnP_module.hparams.opt_alg == 'PnP_SGD':
        if PnP_module.lamb == None:
            PnP_module.lamb = .5
        if PnP_module.std_end == None:
            PnP_module.std = 2. * hparams.noise_level_img /255.
        else:
            PnP_module.std = PnP_module.std_end / 255.
        if PnP_module.stepsize == None:
            PnP_module.stepsize = .8
        if PnP_module.beta == None:
            PnP_module.beta = .01
        if PnP_module.maxitr == None:
            PnP_module.maxitr = 1000

    if hparams.use_wandb:
        if PnP_module.hparams.opt_alg == 'SNORE_Prox' or PnP_module.hparams.opt_alg == 'SNORE':
            PnP_module.hparams.lamb_end = PnP_module.lamb_end = wandb.config.lamb_end
            PnP_module.hparams.std_end = PnP_module.std_end = wandb.config.std_end /255.
        if PnP_module.hparams.opt_alg == 'RED_Prox' or PnP_module.hparams.opt_alg == 'RED':
            PnP_module.hparams.lamb = PnP_module.lamb = wandb.config.lamb
            PnP_module.hparams.sigma_denoiser = PnP_module.sigma_denoiser = wandb.config.std_end /255.
        PnP_module.maxitr = wandb.config.maxitr

     # Set input image paths
    if hparams.image_path is not None : # if a specific image path is given
        input_paths = [hparams.image_path]
        hparams.dataset_name = os.path.splitext(os.path.split(hparams.image_path)[-1])[0]
    else : # if not given, we aply on the whole dataset name given in argument 
        input_path = os.path.join(hparams.dataset_path,hparams.dataset_name)
        input_paths = os_sorted([os.path.join(input_path,p) for p in os.listdir(input_path)])

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
    if PnP_module.hparams.beta != None:
        exp_out_path = os.path.join(exp_out_path, "beta_"+str(PnP_module.hparams.beta))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
    if PnP_module.hparams.sigma_denoiser != None:
        exp_out_path = os.path.join(exp_out_path, "sigma_denoiser_"+str(PnP_module.hparams.sigma_denoiser))
        if not os.path.exists(exp_out_path):
            os.mkdir(exp_out_path)
    if PnP_module.hparams.im_init != None:
        exp_out_path = os.path.join(exp_out_path, "im_init_"+PnP_module.hparams.im_init)
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
        if hparams.grayscale:
            input_im_uint = imread_uint(input_paths[i],n_channels=1)
        else:
            input_im_uint = imread_uint(input_paths[i])
        # if hparams.patch_size < min(input_im_uint.shape[0], input_im_uint.shape[1]):
        #     input_im_uint = crop_center(input_im_uint, hparams.patch_size, hparams.patch_size)
        input_im = np.float32(input_im_uint / 255.)
        # Degraded image
        np.random.seed(0) #for reproductibility
        mask = np.random.binomial(n=1, p=hparams.prop_mask, size=(input_im.shape[0],input_im.shape[1]))
        mask = np.expand_dims(mask,axis=2)
        mask_im = input_im*mask + (0.5)*(1-mask)

        print(hparams.im_init)
        if hparams.im_init == 'random':
            init_im = np.random.random(input_im.shape)
        elif hparams.im_init == 'oracle':
            init_im = input_im
        elif hparams.im_init == 'masked':
            init_im = mask_im
        else:
            init_im = mask_im
        
        # no noise is added
        # np.random.seed(seed=0)
        # mask_im += np.random.normal(0, hparams.noise_level_img/255., mask_im.shape)

        # PnP restoration
        if hparams.extract_images or hparams.extract_curves or hparams.print_each_step:
            inpainted_im, _, output_psnr, output_ssim, output_lpips, output_brisque, output_den_img, output_den_psnr, output_den_ssim, output_den_brisque, output_den_img_tensor, output_den_lpips,_, x_list, z_list, Dg_list, psnr_tab, ssim_tab, brisque_tab, lpips_tab, g_list, F_list, f_list, lamb_tab, std_tab, estimated_noise_list = PnP_module.restore(mask_im, init_im, input_im, mask, extract_results=True)        
        else:
            inpainted_im, _, output_psnr, output_ssim, output_lpips, output_brisque, _, _, _, _, _, _, _ = PnP_module.restore(mask_im, init_im, input_im, mask)

        print('PSNR: {:.2f}dB'.format(output_psnr))
        print('SSIM: {:.2f}'.format(output_ssim))
        psnr_list.append(output_psnr)
        ssim_list.append(output_ssim)
        if not(hparams.grayscale):
            print('LPIPS: {:.2f}'.format(output_lpips))
            print('BRISQUE: {:.2f}'.format(output_brisque))
            lpips_list.append(output_lpips)
            brisque_list.append(output_brisque)
        # psnrY_list.append(output_psnrY)

        if hparams.extract_curves:
            # Create curves
            PnP_module.update_curves(x_list, psnr_tab, ssim_tab, brisque_tab, lpips_tab, Dg_list, g_list, F_list, f_list, lamb_tab, std_tab, estimated_noise_list)

        if hparams.extract_images:
            # Save images
            save_im_path = os.path.join(exp_out_path, 'images')
            if not os.path.exists(save_im_path):
                os.mkdir(save_im_path)

            imsave(os.path.join(save_im_path, 'img_'+str(i)+'_input.png'), input_im_uint)
            imsave(os.path.join(save_im_path, 'img_' + str(i) + "_inpainted.png"), single2uint(np.clip(inpainted_im, 0, 1)))
            imsave(os.path.join(save_im_path, 'img_'+str(i)+'_masked.png'), single2uint(np.clip(mask_im*mask, 0, 1)))
            imsave(os.path.join(save_im_path, 'img_' + str(i) + '_init.png'), single2uint(np.clip(init_im, 0, 1)))

            print('output images saved at ', save_im_path)

            if hparams.save_video:
                save_mov_path = os.path.join(save_im_path, 'img_' + str(i) +"_samples_video")
                fps = 50
                duration = int(1000 * 1 / fps)
                im_list = []
                for x in x_list[::2]:
                    im_list.append(single2uint(rescale(x)))
                imageio.v2.mimsave(save_mov_path+".gif", im_list, duration=duration)

            #save the result of the experiment
            input_im_tensor, masked_im_tensor = array2tensor(input_im).float(), array2tensor(mask_im*mask).float()
            if not(hparams.grayscale):
                dict = {
                        'GT' : input_im,
                        'BRISQUE_GT' : brisque.score(input_im),
                        'estimated_noise_GT' : estimate_sigma(input_im, average_sigmas=True, channel_axis=-1),
                        'Inpainted' : inpainted_im,
                        'Masked' : mask_im*mask,
                        'PSNR_masked' : psnr(input_im, mask_im*mask),
                        'SSIM_masked' : ssim(input_im, mask_im*mask, data_range = 1, channel_axis = 2),
                        'LPIPS_masked' : loss_lpips.forward(input_im_tensor, masked_im_tensor).item(),
                        'BRISQUE_masked' : brisque.score(mask_im*mask),
                        'Init' : mask_im,
                        'SSIM_output' : output_ssim,
                        'PSNR_output' : output_psnr,
                        'LPIPS_output' : output_lpips,
                        'BRISQUE_output' : output_brisque,
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
                        'estimated_noise_list' : estimated_noise_list,
                        'dist_total' : PnP_module.total_dist,
                    }
            else:
                dict = {
                    'GT' : input_im,
                    'Inpainted' : inpainted_im,
                    'Masked' : mask_im*mask,
                    'PSNR_masked' : psnr(input_im, mask_im*mask),
                    'SSIM_masked' : ssim(input_im, mask_im*mask, data_range = 1, channel_axis = 2),
                    'Init' : mask_im,
                    'SSIM_output' : output_ssim,
                    'PSNR_output' : output_psnr,
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
                    'Dg_list' : Dg_list,
                    'g_list' : g_list,
                    'F_list' : F_list,
                    'f_list' : f_list,
                    'lamb_tab' : lamb_tab,
                    'std_tab' : std_tab,
                    'output_den_img' : output_den_img, 
                    'output_den_psnr' : output_den_psnr, 
                    'output_den_ssim' : output_den_ssim,
                    }
            np.save(os.path.join(exp_out_path, 'dict_' + str(i) + '_results'), dict)
        
        if not(hparams.extract_images):
                #save the result of the experiment
                input_im_tensor, masked_im_tensor = array2tensor(input_im).float(), array2tensor(mask_im*mask).float()
                dict = {
                        'GT' : input_im,
                        'BRISQUE_GT' : brisque.score(input_im),
                        'Inpainted' : inpainted_im,
                        'Masked' : mask_im*mask,
                        'PSNR_masked' : psnr(input_im, mask_im*mask),
                        'SSIM_masked' : ssim(input_im, mask_im*mask, data_range = 1, channel_axis = 2),
                        'LPIPS_masked' : loss_lpips.forward(input_im_tensor, masked_im_tensor).item(),
                        'BRISQUE_masked' : brisque.score(mask_im*mask),
                        'Init' : mask_im,
                        'SSIM_output' : output_ssim,
                        'PSNR_output' : output_psnr,
                        'LPIPS_output' : output_lpips,
                        'BRISQUE_output' : output_brisque,
                        'lamb' : PnP_module.lamb,
                        'lamb_0' : PnP_module.lamb_0,
                        'lamb_end' : PnP_module.lamb_end,
                        'maxitr' : PnP_module.maxitr,
                        'std_0' : PnP_module.std_0,
                        'std_end' : PnP_module.std_end,
                        'stepsize' : PnP_module.stepsize,
                        'opt_alg': PnP_module.hparams.opt_alg, 
                    }
                np.save(os.path.join(exp_out_path, 'dict_' + str(i) + '_results'), dict)

        
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

# # # # Start sweep job.
# wandb.agent(sweep_id, function=inpaint)

if __name__ == '__main__' :
    psnr = inpaint()