# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from typing import List

import os
import gc
import copy
from ControlNet.share import *
import einops
import torch
import random

import ControlNet.config as config
from pytorch_lightning import seed_everything
from ControlNet.cldm.model import create_model, load_state_dict
from ControlNet.cldm.ddim_hacked import DDIMSampler

import gradio as gr
import torch
import numpy as np
from zoedepth.utils.arg_utils import parse_unknown
import argparse
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config_user
import gradio as gr

from ui_prediction import predict_depth
import torch.nn.functional as F

from huggingface_hub import hf_hub_download 
import matplotlib

from PIL import Image
import tempfile


def depth_load_state_dict(model, state_dict):
    """Load state_dict into model, handling DataParallel and DistributedDataParallel. Also checks for "model" key in state_dict.

    DataParallel prefixes state_dict keys with 'module.' when saving.
    If the model is not a DataParallel model but the state_dict is, then prefixes are removed.
    If the model is a DataParallel model but the state_dict is not, then prefixes are added.
    """
    state_dict = state_dict.get('model', state_dict)
    # if model is a DataParallel model, then state_dict keys are prefixed with 'module.'

    do_prefix = isinstance(
        model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
    state = {}
    for k, v in state_dict.items():
        if k.startswith('module.') and not do_prefix:
            k = k[7:]

        if not k.startswith('module.') and do_prefix:
            k = 'module.' + k

        state[k] = v

    model.load_state_dict(state, strict=True)
    print("Loaded successfully")
    return model

def load_wts(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    return depth_load_state_dict(model, ckpt)

def load_ckpt(model, checkpoint):
    model = load_wts(model, checkpoint)
    print("Loaded weights from {0}".format(checkpoint))
    return model

def colorize(value, cmap='magma_r', vmin=None, vmax=None):
    percentile = 0.03
    vmin = np.percentile(value, percentile)
    vmax = np.percentile(value, 100 - percentile)

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        value = value * 0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # ((1)xhxwx4)

    value = value[:, :, :3] # bgr -> rgb
    # rgb_value = value[..., ::-1]
    rgb_value = value

    rgb_value = np.transpose(rgb_value, (2, 0, 1))
    rgb_value = rgb_value[np.newaxis, ...]

    return rgb_value

def colorize_depth_maps(depth_map, min_depth=0, max_depth=0, cmap='Spectral_r', valid_mask=None):
    """
    Colorize depth maps.
    """

    percentile = 0.03
    min_depth = np.percentile(depth_map, percentile)
    max_depth = np.percentile(depth_map, 100 - percentile)
    
    assert len(depth_map.shape) >= 2, "Invalid dimension"
    
    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]
    
    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:,:,:,0:3]  # value from 0 to 1
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

def hack_process(path_input, path_depth=None, path_gen=None):
    if path_depth is not None and path_gen is not None:
        return path_input, path_depth, path_gen

def rescale(A, lbound=-1, ubound=1):
    """
    Rescale an array to [lbound, ubound].

    Parameters:
    - A: Input data as numpy array
    - lbound: Lower bound of the scale, default is 0.
    - ubound: Upper bound of the scale, default is 1.

    Returns:
    - Rescaled array
    """
    A_min = np.min(A)
    A_max = np.max(A)
    return (ubound - lbound) * (A - A_min) / (A_max - A_min) + lbound

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, mode, patch_number, resolution_h, resolution_w, patch_size_h, patch_size_w, color_map):
    with torch.no_grad():
        w, h = input_image.size

        detected_map = predict_depth(depth_model, input_image, mode, patch_number, [resolution_h, resolution_w], [patch_size_h, patch_size_w], device=DEVICE, preprocess=preprocess)
        detected_map_save = copy.deepcopy(detected_map)
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        detected_map_save = Image.fromarray((detected_map_save*256).astype('uint16'))
        detected_map_save.save(tmp.name)
        gc.collect()
        torch.cuda.empty_cache() 

        if color_map == 'magma':
            colored_depth = colorize(detected_map)
        else:
            colored_depth = colorize_depth_maps(detected_map) * 255

        detected_map = F.interpolate(torch.from_numpy(detected_map).unsqueeze(dim=0).unsqueeze(dim=0), (image_resolution, image_resolution), mode='bicubic', align_corners=True).squeeze().numpy()

        H, W = detected_map.shape
        detected_map_temp = ((1 - detected_map / (np.max(detected_map + 1e-3))) * 255)
        detected_map = detected_map_temp.astype("uint8")

        detected_map_temp = detected_map_temp[:, :, None]
        detected_map_temp = np.concatenate([detected_map_temp, detected_map_temp, detected_map_temp], axis=2)
        detected_map = detected_map[:, :, None]
        detected_map = np.concatenate([detected_map, detected_map, detected_map], axis=2)
        
        control = torch.from_numpy(detected_map.copy()).float().to(DEVICE) / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255)

        results = [x_samples[i] for i in range(num_samples)]

        return_list = [colored_depth] + results
        update_return_list = []
        for idx, r in enumerate(return_list):
            if idx == 0:
                t_r = torch.from_numpy(r)
            else:
                t_r = torch.from_numpy(r).unsqueeze(dim=0).permute(0, 3, 1, 2)
            # t_r = F.interpolate(t_r, (h, w), mode='bicubic', align_corners=True).squeeze().permute(1, 2, 0).numpy().astype(np.uint8)
            t_r = t_r.squeeze().permute(1, 2, 0).numpy().astype(np.uint8)
            update_return_list.append(t_r)
        update_return_list.append(tmp.name)
        gc.collect()
        torch.cuda.empty_cache()
        
    return update_return_list

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

local_directory = "./nfs"
if not os.path.exists(local_directory):
    os.makedirs(local_directory)

pf_ckp_filename = "patchfusion_u4k.pt"
pf_ckp_filepath = os.path.join(local_directory, pf_ckp_filename)
if not os.path.exists(pf_ckp_filepath):
    print(f"{pf_ckp_filename} not found in local directory. Downloading...")
    pf_ckp_filepath = hf_hub_download(repo_id="zhyever/PatchFusion", filename=pf_ckp_filename, local_dir=local_directory, local_dir_use_symlinks=False)
else:
    print(f"{pf_ckp_filename} found in local directory. Skipping download.")

parser = argparse.ArgumentParser()
parser.add_argument("--ckp_path", type=str, default=pf_ckp_filepath)
ckp_path = pf_ckp_filepath

parser.add_argument("-m", "--model", type=str, default="zoedepth_custom")
model = "zoedepth_custom"

parser.add_argument("--model_cfg_path", type=str, default="./zoedepth/models/zoedepth_custom/configs/config_zoedepth_patchfusion.json")
model_cfg_path = "./zoedepth/models/zoedepth_custom/configs/config_zoedepth_patchfusion.json"

args, unknown_args = parser.parse_known_args()
overwrite_kwargs = parse_unknown(unknown_args)
overwrite_kwargs['model_cfg_path'] = model_cfg_path
overwrite_kwargs["model"] = model
config_depth = get_config_user(model, **overwrite_kwargs)
config_depth["pretrained_resource"] = ''
depth_model = build_model(config_depth)
depth_model = load_ckpt(depth_model, ckp_path)
depth_model.eval().to(DEVICE)

controlnet_ckp_filename = "control_sd15_depth.pth"
controlnet_ckp_filepath = os.path.join(local_directory, controlnet_ckp_filename)
if not os.path.exists(controlnet_ckp_filepath):
    print(f"{controlnet_ckp_filename} not found in local directory. Downloading...")
    controlnet_ckp_filepath = hf_hub_download(repo_id="zhyever/PatchFusion", filename=controlnet_ckp_filename, local_dir=local_directory, local_dir_use_symlinks=False)
else:
    print(f"{controlnet_ckp_filename} found in local directory. Skipping download.")

model = create_model('./ControlNet/models/cldm_v15.yaml')
model.load_state_dict(load_state_dict(controlnet_ckp_filepath, location=DEVICE), strict=False)
model = model.to(DEVICE)
ddim_sampler = DDIMSampler(model)

from zoedepth.models.base_models.midas import Resize
from torchvision.transforms import Compose
preprocess = Compose([Resize(512, 384, keep_aspect_ratio=False, ensure_multiple_of=32, resize_method="minimal")])


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        # local_directory = "./nfs"
        # if not os.path.exists(local_directory):
        #     os.makedirs(local_directory)

        # pf_ckp_filename = "patchfusion_u4k.pt"
        # pf_ckp_filepath = os.path.join(local_directory, pf_ckp_filename)
        # if not os.path.exists(pf_ckp_filepath):
        #     print(f"{pf_ckp_filename} not found in local directory. Downloading...")
        #     pf_ckp_filepath = hf_hub_download(repo_id="zhyever/PatchFusion", filename=pf_ckp_filename, local_dir=local_directory, local_dir_use_symlinks=False)
        # else:
        #     print(f"{pf_ckp_filename} found in local directory. Skipping download.")

        # parser = argparse.ArgumentParser()
        # parser.add_argument("--ckp_path", type=str, default=pf_ckp_filepath)
        # ckp_path = pf_ckp_filepath

        # parser.add_argument("-m", "--model", type=str, default="zoedepth_custom")
        # model = "zoedepth_custom"

        # parser.add_argument("--model_cfg_path", type=str, default="./zoedepth/models/zoedepth_custom/configs/config_zoedepth_patchfusion.json")
        # model_cfg_path = "./zoedepth/models/zoedepth_custom/configs/config_zoedepth_patchfusion.json"

        # args, unknown_args = parser.parse_known_args()
        # overwrite_kwargs = parse_unknown(unknown_args)
        # overwrite_kwargs['model_cfg_path'] = model_cfg_path
        # overwrite_kwargs["model"] = model
        # config_depth = get_config_user(model, **overwrite_kwargs)
        # config_depth["pretrained_resource"] = ''
        # depth_model = build_model(config_depth)
        # depth_model = load_ckpt(depth_model, ckp_path)
        # depth_model.eval().to(DEVICE)

        # controlnet_ckp_filename = "control_sd15_depth.pth"
        # controlnet_ckp_filepath = os.path.join(local_directory, controlnet_ckp_filename)
        # if not os.path.exists(controlnet_ckp_filepath):
        #     print(f"{controlnet_ckp_filename} not found in local directory. Downloading...")
        #     controlnet_ckp_filepath = hf_hub_download(repo_id="zhyever/PatchFusion", filename=controlnet_ckp_filename, local_dir=local_directory, local_dir_use_symlinks=False)
        # else:
        #     print(f"{controlnet_ckp_filename} found in local directory. Skipping download.")

        # model = create_model('./ControlNet/models/cldm_v15.yaml')
        # model.load_state_dict(load_state_dict(controlnet_ckp_filepath, location=DEVICE), strict=False)
        # model = model.to(DEVICE)
        # ddim_sampler = DDIMSampler(model)

        # from zoedepth.models.base_models.midas import Resize
        # from torchvision.transforms import Compose
        # preprocess = Compose([Resize(512, 384, keep_aspect_ratio=False, ensure_multiple_of=32, resize_method="minimal")])

    def predict(
        self,
        input_image: Path = Input(default=None, description="Input image"),
        prompt: str = Input(default='A cozy cottage in an oil painting, with rich textures and vibrant green foliage', description="Prompt"),
        a_prompt: str = Input(default='best quality, extremely detailed', description="Added prompt"),
        n_prompt: str = Input(default='worst quality, low quality, lose details', description="Negative prompt"),
        num_samples: int = Input(default=1, description="Number of images", ge=1, le=1),
        image_resolution: int = Input(default=896, description="ControlNet image resolution", ge=256, le=896),
        ddim_steps: int = Input(default=20, description="Number of steps", ge=1, le=30),
        guess_mode: bool = Input(default=False, description="Guess Mode"),
        strength: float = Input(default=1.0, description="Control strength", ge=0.0, le=2.0),
        scale: float = Input(default=9.0, description="Guidance scale", ge=0.1, le=30.0),
        seed: int = Input(default=-1, description="Seed", ge=-1, le=2147483647),
        eta: float = Input(default=0.0, description="Eta (DDIM)"),
        mode: str = Input(default='P49', description="Tiling mode"),
        patch_number: int = Input(default=256, description="Number of random patches", ge=1, le=256),
        resolution_h: int = Input(default=2160, description="Processing resolution height", ge=256, le=2700),
        resolution_w: int = Input(default=3840, description="Processing resolution width", ge=256, le=4800),
        patch_size_h: int = Input(default=540, description="Patch size height", ge=256, le=675),
        patch_size_w: int = Input(default=960, description="Patch size width", ge=256, le=1200),
        color_map: str = Input(default='magma', description="Colormap used to render depth map"),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        input_image = Image.open(str(input_image))
        update_return_list = process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, mode, patch_number, resolution_h, resolution_w, patch_size_h, patch_size_w, color_map)
        
        print(update_return_list)
        for elem in update_return_list:
            print(type(elem))


        return_list = []
        for i, arr in enumerate(update_return_list):
            if isinstance(arr, np.ndarray):
                # Convert numpy array to PIL Image
                img = Image.fromarray(arr.astype('uint8'))
                # Define the path where you want to save the image
                img_path = Path(f'./output_images/image_{i}.png')
                # Ensure the directory exists
                img_path.parent.mkdir(parents=True, exist_ok=True)
                # Save the image
                img.save(img_path)
                # Append the path to the return list
                return_list.append(img_path)
            # else:
            #     return_list.append(arr)

        return return_list
