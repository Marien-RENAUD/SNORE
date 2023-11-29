python deblur.py --extract_images --extract_curves --noise_level_img 10 --no_backtracking --dataset_name "set4c" --opt_alg "Average_PnP"


# for noise_level_img = 1./255. / take : std_0 = std_end = 1.8 /255. // lamb_end = lamb_0 = 1. // tau = 1 // maxitr = 200
# for noise_level_img = 10./255. / take : std_0 = 18./255. // std_end = 1.8 /255. // lamb_end = lamb_0 = .3 // tau = 1 // maxitr = 600