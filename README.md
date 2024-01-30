# Stochastic deNOising REgularization (SNORE)

Code for the paper "Plug-and-Play image restoration with Stochastic deNOising REgularization"

To develop this code, we use the code from the paper "Gradient Step Denoiser for convergent Plug-and-Play" published at ICLR 2022. [[github](https://github.com/samuro95/GSPnP)] and the code of the library DeepInverse.

This code is made to test in practice for various inverse problems (deblurring, inpainting, super-resolution) the SNORE algorithm and other kind of methods (RED, RED Prox, DiffPIR, PnP-SGD...) and exploring the practice interest of stochastic version of PnP to solve inverse problem in imaging.

## Prerequisites

To installe libraries of correct versions please run the following command
```
pip install -r requirements.txt
```

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

