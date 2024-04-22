python run.py \
    --checkpoint /data_nvme/Depth-Estimation/marigold/checkpoints/marigold-v1-0 \
    --denoise_steps 50 \
    --ensemble_size 10 \
    --input_rgb_dir /data_nvme/Depth-Estimation/booster/train/split_00.txt \
    --output_dir ./testt
    # --output_dir /data_nvme/Depth-Estimation/marigold/output/train_mono \
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
#     --input_rgb_dir /data_nvme/Depth-Estimation/booster/train/split_00.txt \
#     --output_dir /data_nvme/Depth-Estimation/marigold/output/train_mono_lcm

# python run.py \
#     --denoise_steps 4 \
#     --ensemble_size 5 \
#     --input_rgb_dir input/in-the-wild_example \
#     --output_dir output/in-the-wild_example_lcm