for i in 0.1 0.3 1. 1.5 2. 3. 10.
# #i is the number of layers of the denoiser can be between an integer between 1 and 17
do
    python deblur.py --extract_images --lamb $i --extract_curves --gpu_number 1 --kernel_indexes 0 --noise_level_img 10 --dataset_name "set1c" --opt_alg "Average_PnP" --no_backtracking
done

# for noise_level_img = 1./255. / take : std_0 = std_end = 1.8 /255. // lamb_end = lamb_0 = 1. // tau = 1 // maxitr = 200
# for noise_level_img = 10./255. / take : std_0 = 18./255. // std_end = 1.8 /255. // lamb_end = lamb_0 = .3 // tau = 1 // maxitr = 600