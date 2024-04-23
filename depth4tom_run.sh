python run_depth4tom.py \
    --checkpoint /data_nvme/Depth-Estimation/marigold/checkpoints/marigold-v1-0 \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --it 5 \
    --input_rgb_dir /data_nvme/Depth-Estimation/booster/train/split_rgb_and_mask_00.txt \
    --output_dir /data_nvme/Depth-Estimation/marigold/output/train_mono_50_10_5 \
    # --output_dir ./testt
    # --denoise_variance

# 159 images
# /data_nvme/Depth-Estimation/booster/test_mono_nogt/split.txt
# 456 images
# /data_nvme/Depth-Estimation/booster/train/split.txt
# /data_nvme/Depth-Estimation/booster/val_mono_nogt/split.txt

# python run.py \
#     --checkpoint /data_nvme/Depth-Estimation/marigold/checkpoints/marigold-lcm-v1-0 \
#     --denoise_steps 4 \
#     --ensemble_size 5 \
#     --it 5 \
#     --input_rgb_dir /data_nvme/Depth-Estimation/booster/train/split_rgb_and_mask_00.txt \
#     --output_dir /data_nvme/Depth-Estimation/marigold/output/train_mono_lcm

# python run.py \
#     --denoise_steps 4 \
#     --ensemble_size 5 \
#     --input_rgb_dir input/in-the-wild_example \
#     --output_dir output/in-the-wild_example_lcm