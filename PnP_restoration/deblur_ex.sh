for i in 0.01 0.05 0.07 0.09 0.1 0.5 0.7 
# #i is the number of layers of the denoiser can be between an integer between 1 and 17
do
    python deblur.py --extract_images --extract_curves --lamb $i --gpu_number 0 --kernel_indexes 0 --noise_level_img 10 --dataset_name "set1c" --opt_alg "PnP_GD"
done

# for noise_level_img = 1./255. / take : std_0 = std_end = 1.8 /255. // lamb_end = lamb_0 = 1. // tau = 1 // maxitr = 200
# for noise_level_img = 10./255. / take : std_0 = 18./255. // std_end = 1.8 /255. // lamb_end = lamb_0 = .3 // tau = 1 // maxitr = 600