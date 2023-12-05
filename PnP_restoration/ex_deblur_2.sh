for i in 0.25 0.3 0.35 0.4 0.45 0.5
# #i is the number of layers of the denoiser can be between an integer between 1 and 17
do
    python deblur.py --extract_images --lamb $i --extract_curves --gpu_number 1 --kernel_indexes 0 --noise_level_img 10 --dataset_name "set1c" --opt_alg "PnP_GD"
done