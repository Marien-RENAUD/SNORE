for i in 0.05 0.1 0.15
# #i is the number of layers of the denoiser can be between an integer between 1 and 17
do
    python deblur.py --extract_images --no_backtracking --lamb $i --extract_curves --gpu_number 0 --kernel_indexes 0 --noise_level_img 20 --dataset_name "set1c" --opt_alg "PnP_Prox"
done

# for noise_level_img = 1./255. / take : std_0 = std_end = 1.8 /255. // lamb_end = lamb_0 = 1. // tau = 1 // maxitr = 200
# for noise_level_img = 10./255. / take : std_0 = 18./255. // std_end = 1.8 /255. // lamb_end = lamb_0 = .3 // tau = 1 // maxitr = 600