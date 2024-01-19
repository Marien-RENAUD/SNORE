# for i in 0.1 0.5 1. 1.5 10.
# do
#         python deblur.py --lamb_end $i --extract_images --extract_curves --gpu_number 1 --kernel_indexes 0 --noise_level_img 10 --dataset_name "totem" --opt_alg "Average_PnP"
# done

# for i in 0.01 0.05 0.1 0.15 1.
# do
#         python deblur.py --lamb_0 $i --extract_images --extract_curves --gpu_number 1 --kernel_indexes 0 --noise_level_img 10 --dataset_name "totem" --opt_alg "Average_PnP"
# done

for i in 0.0196 0.0392 0.0706 0.1176 #5 10 18 30
do
        python deblur.py --std_0 $i --extract_images --extract_curves --gpu_number 1 --kernel_indexes 0 --noise_level_img 10 --dataset_name "totem" --opt_alg "Average_PnP"
done

for i in 0.00392 0.0117 0.0196 0.0392 0.0706 #1 3 5 10 18
do
        python deblur.py --std_end $i --extract_images --extract_curves --gpu_number 1 --kernel_indexes 0 --noise_level_img 10 --dataset_name "totem" --opt_alg "Average_PnP"
done
# see different options of parser argument at the end of GS_PnP_restoration.py