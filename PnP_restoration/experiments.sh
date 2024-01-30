# For deblurring with SNORE
python deblur.py --dataset_name "CBSD10" --opt_alg "SNORE" --noise_level_img 10. --kernel_indexes 0 --extract_images --extract_curves

# For inpainting with RED_Prox
python inpaint.py --dataset_name "CBSD10" --opt_alg "RED_Prox" --extract_images --extract_curves

# For deblurring with DiffPIR
python DiffPIR.py --dataset_name "CBSD10" --degradation "deblurring"