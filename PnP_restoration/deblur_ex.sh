for i in 1 2 3 4 5 6 7 8 9 10 
do
# 11 12 13 14 15 16 17 18 19 20 21 22
# #i is the number of layers of the denoiser can be between an integer between 1 and 17
    python deblur.py --annealing_number $i --extract_images --extract_curves --gpu_number 0 --kernel_indexes 0 --noise_level_img 10 --dataset_name "set4c" --opt_alg "Average_PnP"
done

# for noise_level_img = 1./255. / take : std_0 = std_end = 1.8 /255. // lamb_end = lamb_0 = 1. // tau = 1 // maxitr = 200
# for noise_level_img = 10./255. / take : std_0 = 18./255. // std_end = 1.8 /255. // lamb_end = lamb_0 = .3 // tau = 1 // maxitr = 600