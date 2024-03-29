import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
import os
import argparse
import cv2
import imageio


source_dataset_path = "../datasets/SAR_test/SAR"

clock = 0

for file in os.listdir(source_dataset_path):
    filename = os.fsdecode(file)
    im = cv2.imread(source_dataset_path+"/"+filename)
    print(im.shape)
    n, m, _ = im.shape
    for i in tqdm(range(0,n-256,128)):
        for j in range(0,m-256,128):
            im_crop = im[i:i+256, j:j+256, :]
            plt.imsave("../datasets/SAR_test/"+str(clock)+".png", im_crop)
            clock += 1
