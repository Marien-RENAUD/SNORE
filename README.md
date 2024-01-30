# Stochastic deNOising REgularization (SNORE)

Code for the paper "Plug-and-Play image restoration with Stochastic deNOising REgularization"

To develop this code, we use the code from the paper "Gradient Step Denoiser for convergent Plug-and-Play" published at ICLR 2022. [[github](https://github.com/samuro95/GSPnP)] and the code of the library DeepInverse.

This code is made to test in practice for various inverse problems (deblurring, inpainting, super-resolution) the SNORE algorithm and other kind of methods (RED, RED Prox, DiffPIR, PnP-SGD...) and exploring the practice interest of stochastic version of PnP to solve inverse problem in imaging.

## Prerequisites

To installe libraries of correct versions please run the following command
```
pip install -r requirements.txt
```
