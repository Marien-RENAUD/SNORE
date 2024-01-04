for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
do
    python deblur.py --seed $i --extract_images --extract_curves --gpu_number 1 --kernel_indexes 0 --noise_level_img 5 --dataset_name "set1c" --opt_alg "Average_PnP"
done
# see different options of parser argument at the end of GS_PnP_restoration.py