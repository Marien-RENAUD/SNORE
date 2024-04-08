# Stochastic deNOising REgularization (SNORE)

Here are videos of SNORE iterative process for image deblurring and inpainting.
<table>
  <tr>
    <td><img src="https://github.com/Marien-RENAUD/SNORE/blob/main/images/img_0_samples_video_deb.gif" width="150" height="225" loop=infinite/></td>
    <td><img src="https://github.com/Marien-RENAUD/SNORE/blob/main/images/img_1_samples_video_deb.gif" width="150" height="225" loop=infinite/></td>
    <td><img src="https://github.com/Marien-RENAUD/SNORE/blob/main/images/img_0_samples_video_imp.gif" width="150" height="225" loop=infinite/></td>
    <td><img src="https://github.com/Marien-RENAUD/SNORE/blob/main/images/img_1_samples_video_imp.gif" width="150" height="225" loop=infinite/></td>
  </tr>
</table>



Code for the paper ["Plug-and-Play image restoration with Stochastic deNOising REgularization"](https://arxiv.org/abs/2402.01779)

To develop this code, we use the code from the paper "Gradient Step Denoiser for convergent Plug-and-Play" published at ICLR 2022. [[github](https://github.com/samuro95/GSPnP)] and the code of the library DeepInverse.

This code is made to test in practice for various inverse problems (deblurring, inpainting, super-resolution) the SNORE algorithm and other kind of methods (RED, RED Prox, DiffPIR, PnP-SGD...) and exploring the practice interest of stochastic version of PnP to solve inverse problem in imaging.

## Prerequisites

To installe libraries of correct versions please run the following command
```
pip install -r requirements.txt
```

Download pretrained checkpoint https://plmbox.math.cnrs.fr/f/ab6829cb933743848bef/?dl=1 for color denoising and save it as ```GS_denoising/ckpts/GSDRUNet.ckpt```

## Experiments

Example of experiments are provided in the file PnP_restoration/experiments.sh.

All experiments results are saved in a folder outside the SNORE repository named "Result_SNORE".

### Deblurring

For image deblurring, one can run for instance the following command
```
python deblur.py --dataset_name "CBSD10" --opt_alg "SNORE" --noise_level_img 10. --kernel_indexes 0 --extract_images --extract_curves
```
It compute the SNORE restoration on images of CBSD10 dataset with the kernel 0 and an input noise level of $\sigma_{\mathbf{y}} = 10/255$. "extract_images" and "extract_images" save visual results including optimization curves, annealing parameters curves and clean, degraded and output images.

![Deblurring of various images with various technics including SNORE](images/set_of_results_deblurring.png)


### Inpainting

For image inpainting, one can run for instance the following command
```
python inpaint.py --dataset_name "CBSD10" --opt_alg "SNORE" --extract_images --extract_curves
```
It compute the SNORE restoration on images of CBSD10 dataset with a mask of probability 0.5. "extract_images" and "extract_images" save visual results including optimization curves, annealing parameters curves and clean, degraded and output images.

![Inpainting of various images with various technics including SNORE](images/set_of_results_inpainting.png)

## File Structure
```
- datasets : collection of used datasets
- GS_denoising : code to define the Gradient-step denoiser
- images : images for the README.md
- PnP_restoration : code for restoration
  - kernels : collection of kernels used for deblurring
  - utils : some useful functions and settings
  - deblur.py : code for image deblurring with various methods including SNORE, SNORE_Prox, RED, RED_Prox
  - inpaint.py : code for image inpainting with various methods including SNORE, SNORE_Prox, RED, RED_Prox
  - DiffPIR.py : code for image restoration including deblurring and inpainting with DiffPIR. This implementation is inspired by the code of the Python library DeepInverse.
  - GS_PnP_restoration.py : code to compute the optimization with various methods. Note that the optimization process is coded in this file. SNORE, SNORE_Prox, RED, RED_Prox have been qualify. We also provide code for SNORE_Adam (SNORE regularization optimized with ADAM), ARED_Prox (RED_Prox with an annealing procedure) and PnP_SGD (a stochastic gradient descent with a PnP regularization). However, these technics has not been qualified and studied in details, so it is possible that this implementation is not correct.
  - SR.py : code for image super-resolution. Note that the current code is only adapted for SNORE Prox and RED Prox.
```

## Parameters
### Common parameter
To use a parameters that is list below, here is some examples of command
```
python inpaint.py --dataset_name "CBSD10" --opt_alg "RED Prox" --lamb 0.1 --stepsize 10. --no_backtracking --no_early_stopping
python deblur.py --dataset_name "set1c" --opt_alg "RED" --lamb 0.3 --sigma_denoiser 5.
python SR.py --dataset_name "set3c" --opt_alg "PnP SGD" --beta 0.02 --lamb 0.5 --save_video --extract_curves --extract_images
```

- opt_alg : to choose the optimization algorithm, we implement Stochastique Denoising Regularization 'SNORE', Stochastique Denoising Regularization with a proximal data-fidelity 'SNORE Prox', ADAM algorithm apply on SNORE 'SNORE_Adam', a gradient descent without regularization 'Data_GD', Regularization by Denoising 'RED', Regularization by Denoising with a proximal data-fidelity 'RED Prox', Annealed Regularization by Denoising with a proximal data-fidelity 'ARED Prox', Plug-and-Play Stochastic Gradient Descent 'PnP_SGD'
- dataset_path : the path of used datasets, by defaults the provided datasets in '../datasets'
- gpu_number' : the index of the used gpu, by default 0
- pretrained_checkpoint : weights of the GS Drunet denoiser, by default they are saved in the path '../GS_denoising/ckpts/GSDRUNet.ckpt'
- im_init : to change the initialization, by default it is the observation of deblurring and inpainting (with value 0.5 on masked pixels). Can be change to "random" for a random initialization or "oracle" for the true image. This parameter is implemented for deblurring and inapinting.
- noise_model : the noise model, by default 'gaussian', the 'speckle' noise model is also implemented.
- dataset_name : the path of the dataset restore by our algorithm, by default 'set3c'
- noise_level_img : the noise level of the observation
- maxitr : to change the default number of iteration of the used algorithm
- stepsize : to change the default stepsize of the used algorithm
- lamb : to change the default regularization parameter
- beta : to change the default parameter to control the amount of added noise in PnP SGD
- std_0 : to change the default initial denoiser parameter for SNORE or SNORE Prox (with annealing)
- std_end : to change the default final denoiser parameter for SNORE or SNORE Prox (with annealing)
- lamb_0 : to change the default initial regularization parameter for SNORE or SNORE Prox (with annealing)
- lamb_end : to change the default final regularization parameter for SNORE or SNORE Prox (with annealing)
- num_noise : the number of random sample for the stochastic gradient approximation, increasing num_noise reduce the variance of the estimator and increse the computation cost, by default 1
- annealing_number : the number of annealing level for SNORE or SNORE Prox, by default 16
- last_itr : the number of final iterations with the final annealing parameters for SNORE or SNORE Prox
- sigma_denoiser : to change the denoiser parameter for non-annealing algorithm
- seed : to define a random seed to the sample during stochastic algorithm
- lpips : to compute the lpips for each iteration, computationally expensive
- no_backtracking : to cancel the use of backtracking for RED or RED Prox
- extract_curves : to save curves of iterations, PSNR, SSIM, BRISQUE, F, f, g... during iterations
- extract_images : to save images
- save_video : to save a video of the iterations
- print_each_step : to print at each step the optimization function value, the current PSNR, SSIM
- no_data_term : to only optimize on the regularization
- use_hard_constraint : projected at each step the restore image in [0,1]
- use_wandb : use the library weights and biais
- grayscale : to use a grayscale denoiser
- no_early_stopping : to not use early stop for RED or RED Prox
- exp_out_path : the path of output saving, by default create a folder outside the project in the path "../../Result_SNORE"

### Deblurring
- kernel_path : to use it own kernel of blur save at numpy array
- kernel_indexes : the chose the used kernel for restoration integer between 0 and 9, by default all kernel are applied and restored successively

### Inpainting
- prop_mask : to change the masked pixels proportion, by default 0.5

### Super-resolution
- sf : to change the super-resolution factor, by default 2
- kernel_path : to use it own kernel of blur save at numpy array
- kernel_indexes : the chose the used kernel for restoration integer between 0 and 9, by default all kernel are applied and restored successively

### Despeckling
- L : to change the number of look, by default 20





