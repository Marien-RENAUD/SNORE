for i in 0 1 2 3 4 5 6 7 8 9
do
    python test_deepinv.py --kernel_index $i --dataset_name "CBSD10" --noise_level_img 20. --gpu_number 0
done