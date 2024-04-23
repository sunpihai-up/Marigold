# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import argparse
import logging
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

import matplotlib

from marigold import MarigoldPipeline

EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

def get_median(pred_list):
    stacked_array = np.stack(pred_list)
    return np.median(stacked_array, axis=0)

def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored

def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-lcm-v1-0",
        help="Checkpoint path or hub name.",
    )

    parser.add_argument(
        "--input_rgb_dir",
        type=str,
        required=True,
        help="Path to the input image folder.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=4,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed. For the original (DDIM) version, it's recommended to use 10-50 steps, while for LCM 1-4 steps.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`. Default: `bilinear`",
    )

    # depth map colormap
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render depth predictions.",
    )
    
    parser.add_argument(
        "--denoise_variance",
        action="store_true",
        help="Calculate the variance between denoising steps.",
    )

    # other settings
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Reproducibility seed. Set to `None` for unseeded inference.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--apple_silicon",
        action="store_true",
        help="Flag of running on Apple Silicon.",
    )
    
    parser.add_argument('--it', 
        default=1,
        type=int,
        help="number of iteration to compute depth"
    )

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    input_rgb_dir = args.input_rgb_dir
    output_dir = args.output_dir
    it = args.it

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    color_map = args.color_map
    denoise_variance = args.denoise_variance
    seed = args.seed
    batch_size = args.batch_size
    apple_silicon = args.apple_silicon
    if apple_silicon and 0 == batch_size:
        batch_size = 1  # set default batchsize

    # -------------------- Preparation --------------------
    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps}, ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res}, seed = {seed}; "
        f"color_map = {color_map}."
    )

    # Output directories
    output_dir_color = os.path.join(output_dir, "depth_colored")
    output_dir_tif = os.path.join(output_dir, "depth_bw")
    output_dir_npy = os.path.join(output_dir, "depth_npy")
    output_dir_uncertainty_npy = os.path.join(output_dir, "uncertainty_npy")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_tif, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    os.makedirs(output_dir_uncertainty_npy, exist_ok=True)
    
    if denoise_variance:
        output_dir_variance_npy = os.path.join(output_dir, "variance_npy")
        os.makedirs(output_dir_variance_npy, exist_ok=True)
    
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if apple_silicon:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps:0")
        else:
            device = torch.device("cpu")
            logging.warning("MPS is not available. Running on CPU will be slow.")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    assert input_rgb_dir[-3:] == 'txt', f"{input_rgb_dir} is not a valid file path"
    
    # Get rgb path from split file
    rgb_filename_list = []
    mask_filename_list = []
    with open(input_rgb_dir, 'r') as f:
        for line in f:
            line = line.rstrip()
            rgb_path, mask_path = line.split()[0], line.split()[1]
            
            if not os.path.isfile(rgb_path):
                print(f"{rgb_path} is missed!")
            else:
                rgb_filename_list.append(rgb_path)
            
            if not os.path.isfile(mask_path):
                print(f"{mask_path} is missed!")
            else:
                mask_filename_list.append(mask_path)
                
    n_images = len(rgb_filename_list)
    if n_images > 0:
        logging.info(f"Found {n_images} images")
    else:
        logging.error(f"No image found in '{input_rgb_dir}'")
        exit(1)

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        variant = "fp16"
        logging.info(
            f"Running with half precision ({dtype}), might lead to suboptimal result."
        )
    else:
        dtype = torch.float32
        variant = None

    pipe = MarigoldPipeline.from_pretrained(
        checkpoint_path, variant=variant, torch_dtype=dtype
    )

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass  # run without xformers

    pipe = pipe.to(device)

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)

        # for rgb_path in tqdm(rgb_filename_list, desc="Estimating depth", leave=True):
        for i in tqdm(range(len(rgb_filename_list)), desc="Estimating depth", leave=True):
            rgb_path = rgb_filename_list[i]
            mask_path = mask_filename_list[i]
            
            # Read input image and mask image
            input_image = Image.open(rgb_path)
            mask_image = Image.open(mask_path)
            
            # Convert Image to NDArray
            rgb_array = np.array(input_image)
            mask_array = np.array(mask_image)

            depth_pred_list = []
            uncertainty_map_list = []
            if denoise_variance:
                variance_heat_map_list = []
            
            # for _ in range(it):
            for _ in tqdm(range(it), desc = "Iterative color", leave=False):          
                # Randomly generate colors
                random_color = np.random.randint(0, 256, size=3)
                
                # Create a new image array for inpainting
                painted_rgb = np.copy(rgb_array)
                
                # Replace pixels with a mask value of 255 with random colors
                painted_rgb[mask_array == 255] = random_color
                
                # Convert NDArray to Image
                painted_rgb_image = Image.fromarray(painted_rgb)
                # Predict depth
                pipe_out = pipe(
                    painted_rgb_image,
                    denoising_steps=denoise_steps,
                    ensemble_size=ensemble_size,
                    processing_res=processing_res,
                    match_input_res=match_input_res,
                    batch_size=batch_size,
                    color_map=None,
                    denoise_variance=denoise_variance,
                    show_progress_bar=True,
                    resample_method=resample_method,
                    seed=seed,
                )

                depth_pred: np.ndarray = pipe_out.depth_np
                depth_pred_list.append(depth_pred)
                                
                # Uncertainty map between ensemble
                uncertainty_map: np.ndarray = pipe_out.uncertainty
                uncertainty_map_list.append(uncertainty_map)
                
                if denoise_variance:
                    variance_heat_map: np.ndarray = pipe_out.variance_heat_map
                    variance_heat_map_list.append(variance_heat_map)
            
            depth_pred = get_median(depth_pred_list)
            uncertainty_map = get_median(uncertainty_map_list)
            if denoise_variance:
                variance_heat_map = get_median(variance_heat_map_list)
            
            # Colorize Depth Map
            if color_map is not None:
                depth_colored = colorize_depth_maps(
                    depth_pred, 0, 1, cmap=color_map
                ).squeeze()  # [3, H, W], value in (0, 1)
                depth_colored = (depth_colored * 255).astype(np.uint8)
                depth_colored_hwc = chw2hwc(depth_colored)
                depth_colored_img = Image.fromarray(depth_colored_hwc)
            else:
                depth_colored_img = None
            
            # TODO: 保存名称加上摄像机编号
            # Save as npy
            dir_name = rgb_path.split('/')[5]
            rgb_name_base = dir_name + '_' + os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base + "_pred"
            
            # Depth Map
            npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
            if os.path.exists(npy_save_path):
                logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
            np.save(npy_save_path, depth_pred)
            
            # Uncertainty Map
            npy_save_path = os.path.join(output_dir_uncertainty_npy, f"{pred_name_base}_uncertainty.npy")
            if os.path.exists(npy_save_path):
                logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
            np.save(npy_save_path, uncertainty_map)

            # Variance Map
            if denoise_variance:
                npy_save_path = os.path.join(output_dir_variance_npy, f"{pred_name_base}_variance.npy")
                if os.path.exists(npy_save_path):
                    logging.warning(f"Existing file: '{npy_save_path}' will be overwritten")
                np.save(npy_save_path, variance_heat_map)
            
            # # Save as 16-bit uint png
            # depth_to_save = (depth_pred * 65535.0).astype(np.uint16)
            # png_save_path = os.path.join(output_dir_tif, f"{pred_name_base}.png")
            # if os.path.exists(png_save_path):
            #     logging.warning(f"Existing file: '{png_save_path}' will be overwritten")
            # Image.fromarray(depth_to_save).save(png_save_path, mode="I;16")

            # Colorize Depth
            colored_save_path = os.path.join(
                output_dir_color, f"{pred_name_base}_colored.png"
            )
            if os.path.exists(colored_save_path):
                logging.warning(
                    f"Existing file: '{colored_save_path}' will be overwritten"
                )
            depth_colored_img.save(colored_save_path)