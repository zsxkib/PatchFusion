# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Zhenyu Li

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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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


from huggingface_hub import hf_hub_download

# Specify your desired local directory path
local_directory = "./nfs"
if not os.path.exists(local_directory):
    os.makedirs(local_directory)

# pf_ckp = hf_hub_download(repo_id="zhyever/PatchFusion", filename="patchfusion_u4k.pt")
# pf_ckp = hf_hub_download(repo_id="zhyever/PatchFusion", filename="patchfusion_u4k.pt", local_dir=local_directory, local_dir_use_symlinks=False)
pf_ckp_filename = "patchfusion_u4k.pt"
pf_ckp_filepath = os.path.join(local_directory, pf_ckp_filename)
if not os.path.exists(pf_ckp_filepath):
    print(f"{pf_ckp_filename} not found in local directory. Downloading...")
    pf_ckp_filepath = hf_hub_download(repo_id="zhyever/PatchFusion", filename=pf_ckp_filename, local_dir=local_directory, local_dir_use_symlinks=False)
else:
    print(f"{pf_ckp_filename} found in local directory. Skipping download.")

parser = argparse.ArgumentParser()
parser.add_argument("--ckp_path", type=str, default=pf_ckp_filepath)
parser.add_argument("-m", "--model", type=str, default="zoedepth_custom")
parser.add_argument("--model_cfg_path", type=str, default="./zoedepth/models/zoedepth_custom/configs/config_zoedepth_patchfusion.json")
args, unknown_args = parser.parse_known_args()
overwrite_kwargs = parse_unknown(unknown_args)
overwrite_kwargs['model_cfg_path'] = args.model_cfg_path
overwrite_kwargs["model"] = args.model
config_depth = get_config_user(args.model, **overwrite_kwargs)
config_depth["pretrained_resource"] = ''
depth_model = build_model(config_depth)
depth_model = load_ckpt(depth_model, args.ckp_path)
depth_model.eval().to(DEVICE)

# controlnet_ckp = hf_hub_download(repo_id="zhyever/PatchFusion", filename="control_sd15_depth.pth")
# controlnet_ckp = hf_hub_download(repo_id="zhyever/PatchFusion", filename="control_sd15_depth.pth", local_dir=local_directory, local_dir_use_symlinks=False)
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

title = "# PatchFusion"
description = """Official demo for **PatchFusion: An End-to-End Tile-Based Framework for High-Resolution Monocular Metric Depth Estimation**.

PatchFusion is a deep learning model for high-resolution metric depth estimation from a single image.

Please refer to our [project webpage](https://zhyever.github.io/patchfusion), [paper](https://arxiv.org/abs/2312.02284) or [github](https://github.com/zhyever/PatchFusion) for more details.

# Advanced tips

The overall pipeline: image --> (PatchFusion) --> depth --> (controlnet) --> generated image.

As for the PatchFusion, it works on default 4k (2160x3840) resolution. All input images will be resized to 4k before passing through PatchFusion as default. It means if you have a higher resolution image, you might want to increase the processing resolution in the advanced option (You would also change the patch size to 1/4 image resolution!). Because of the tiling strategy, our PatchFusion would not use more memory or time for even higher resolution inputs if properly setting parameters. 

The output depth map is resized to the original image resolution. Download for better visualization quality. 16-Bit Raw Depth = (pred_depth * 256).to(uint16).

We provide two color maps to render depth map, which are magma (more common in supervised depth estimation) and spectral (better looking). Please choose from the advanced option.

For ControlNet, it works on default 896x896 resolution. Again, all input images will be resized to 896x896 before passing through ControlNet as default. You might be not happy because the 4K->896x896 downsampling, but limited by the GPU resource, this demo could only achieve this. This is the memory bottleneck. The output is not resized back to the image resolution for fast inference (Well... It's still so slow now... :D).

We provide some tips might be helpful: (1) Try our experimental demo (check our github) running on a local 80G gpu (you could try high-resolution generation there, like the one in our paper). But of course, it would be expired soon (in two days maybe); (2) Clone our code repo, and look for a gpu with more than 24G memory; (3) Clone our code repo, run the depth estimation (there are another demos for depth estimation and image-to-3D), and search for another guided high-resolution image generation strategy; (4) Some kind people give this space a stronger gpu support.

NOTE: the overall inference time of P49 PatchFusion and 20-step diffusion is about 60s.
"""

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Depth Maps")
    
    with gr.Row():
        with gr.Accordion("Advanced options", open=False):
            # mode = gr.Radio(["P49", "R"], label="Tiling mode", info="We recommand using P49 for fast evaluation and R with 1024 patches for best visualization results, respectively", elem_id='mode', value='R')
            mode = gr.Radio(["P49", "R"], label="(PatchFusion) Tiling mode", info="We recommand using P49 for fast evaluation and R with 256 patches for best visualization results, respectively", elem_id='mode', value='P49')
            patch_number = gr.Slider(1, 256, label="(PatchFusion) Please decide the number of random patches (Only useful in mode=R)", step=1, value=256)
            # resolution = gr.Textbox(label="(PatchFusion) Proccessing resolution (Default 4K. Use 'x' to split height and width.)", elem_id='mode', value='2160x3840')
            # patch_size = gr.Textbox(label="(PatchFusion) Patch size (Default 1/4 of image resolution. Use 'x' to split height and width.)", elem_id='mode', value='540x960')
            resolution_h = gr.Slider(label="(PatchFusion) Proccessing resolution height (Default 4K, 2160)", minimum=256, maximum=2700, value=2160, step=1)
            resolution_w = gr.Slider(label="(PatchFusion) Proccessing resolution width (Default 4K, 3840)", minimum=256, maximum=4800, value=3840, step=1)
            patch_size_h = gr.Slider(label="(PatchFusion) Patch size height (Default 4K x 1/4, 540)", minimum=256, maximum=675, value=540, step=1)
            patch_size_w = gr.Slider(label="(PatchFusion) Patch size width (Default 4K x 1/4, 960)", minimum=256, maximum=1200, value=960, step=1)
        
            color_map = gr.Radio(["magma", "spectral"], label="(PatchFusion) Colormap used to render depth map", elem_id='mode', value='magma')

            # num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
            num_samples = gr.Slider(label="(ControlNet) Images", minimum=1, maximum=1, value=1, step=1)
            image_resolution = gr.Slider(label="(ControlNet) ControlNet image resolution (higher resolution will lead to OOM)", minimum=256, maximum=896, value=896, step=64)
            strength = gr.Slider(label="(ControlNet) Control strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
            guess_mode = gr.Checkbox(label='(ControlNet) Guess Mode', value=False)
            # detect_resolution = gr.Slider(label="Depth Resolution", minimum=128, maximum=1024, value=384, step=1)
            ddim_steps = gr.Slider(label="(ControlNet) steps", minimum=1, maximum=30, value=20, step=1)
            scale = gr.Slider(label="(ControlNet) guidance scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
            seed = gr.Slider(label="(ControlNet) seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
            eta = gr.Number(label="(ControlNet) eta (DDIM)", value=0.0)
            a_prompt = gr.Textbox(label="(ControlNet) Added prompt", value='best quality, extremely detailed')
            n_prompt = gr.Textbox(label="(ControlNet) Negative prompt", value='worst quality, low quality, lose details')

    with gr.Row():
        with gr.Column():
            # input_image = gr.Image(source='upload', type="pil")
            input_image = gr.Image(label="Input Image", type='pil')
            prompt = gr.Textbox(label="Prompt (input your description)", value='A cozy cottage in an oil painting, with rich textures and vibrant green foliage')
            run_button = gr.Button("Run")
            
        generated_image = gr.Image(label="Generated Map", elem_id='img-display-output')

    with gr.Row():
        depth_image = gr.Image(label="Depth Map", elem_id='img-display-output')
    with gr.Row():
        raw_file = gr.File(label="16-Bit Raw Depth, Multiplier:256")

    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, mode, patch_number, resolution_h, resolution_w, patch_size_h, patch_size_w, color_map]
    run_button.click(fn=process, inputs=ips, outputs=[depth_image, generated_image, raw_file])
    examples = gr.Examples(
        inputs=[input_image, depth_image, generated_image],
        outputs=[input_image, depth_image, generated_image],
        examples=[
            [
                "examples/example_4.jpeg",
                "examples/2_depth.png",
                "examples/2_gen.png",

            ],
            [
                "examples/example_6.png",
                "examples/4_depth.png",
                "examples/4_gen.png",
            ],
            [
                "examples/example_1.jpeg",
                "examples/1_depth.png",
                "examples/1_gen.png",
            ],],
        cache_examples=True,
        fn=hack_process)

if __name__ == '__main__':
    demo.queue().launch(share=True)