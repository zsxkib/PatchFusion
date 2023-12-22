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
import cv2
import argparse
from zoedepth.utils.config import get_config_user
from zoedepth.models.builder import build_model
from zoedepth.utils.arg_utils import parse_unknown
import numpy as np
from zoedepth.models.base_models.midas import Resize
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
from zoedepth.utils.misc import get_boundaries
from zoedepth.utils.misc import compute_metrics, RunningAverageDict
from tqdm import tqdm
import matplotlib
import torch.nn.functional as F
from zoedepth.data.middleburry import readPFM
import random
import imageio
from PIL import Image

def load_state_dict(model, state_dict):
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
    # model.load_state_dict(state, strict=False)
    print("Loaded successfully")
    return model


def load_wts(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    return load_state_dict(model, ckpt)

def load_ckpt(model, checkpoint):
    model = load_wts(model, checkpoint)
    print("Loaded weights from {0}".format(checkpoint))
    return model

#### def dataset
def read_image(path, dataset_name):
    if dataset_name == 'u4k':
        img = np.fromfile(open(path, 'rb'), dtype=np.uint8).reshape(2160, 3840, 3) / 255.0
        img = img.astype(np.float32)[:, :, ::-1].copy()
    elif dataset_name == 'mid':
        img = cv2.imread(path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        
        img = F.interpolate(torch.tensor(img).unsqueeze(dim=0).permute(0, 3, 1, 2), IMG_RESOLUTION, mode='bicubic', align_corners=True)
        img = img.squeeze().permute(1, 2, 0)
    
    elif dataset_name == 'nyu':
        img = Image.open(path)
        img = np.asarray(img, dtype=np.float32) / 255.0
        
    else:
        img = cv2.imread(path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        print(img.shape)
        img = F.interpolate(torch.tensor(img).unsqueeze(dim=0).permute(0, 3, 1, 2), IMG_RESOLUTION, mode='bicubic', align_corners=True)
        img = img.squeeze().permute(1, 2, 0)
    return img

class Images:
    def __init__(self, root_dir, files, index, dataset_name=None):
        self.root_dir = root_dir
        name = files[index]
        self.dataset_name = dataset_name
        self.rgb_image = read_image(os.path.join(self.root_dir, name), dataset_name)
        name = name.replace(".jpg", "")
        name = name.replace(".png", "")
        name = name.replace(".jpeg", "")
        self.name = name
        
class DepthMap:
    def __init__(self, root_dir, files, index, dataset_name, pred=False):
        self.root_dir = root_dir
        name = files[index]

        gt_path = os.path.join(self.root_dir, name)
        if dataset_name == 'u4k':
            depth_factor = gt_path.replace('val_gt', 'val_factor')
            depth_factor = depth_factor.replace('.npy', '.txt')
            with open(depth_factor, 'r') as f:
                df = f.readline()
            df = float(df)

            gt_disp = np.load(gt_path, mmap_mode='c')
            gt_disp = gt_disp.astype(np.float32)
            edges = get_boundaries(gt_disp, th=1, dilation=0)

            gt_depth = df/gt_disp
            self.gt = gt_depth
            self.edge = edges

        elif dataset_name == 'gta':
            gt_depth = imageio.imread(gt_path)
            gt_depth = np.array(gt_depth).astype(np.float32) / 256
            edges = get_boundaries(gt_depth, th=1, dilation=0)
            self.gt = gt_depth
            self.edge = edges
        
        elif dataset_name == 'mid':
            
            depth_factor = gt_path.replace('gts', 'calibs')
            depth_factor = depth_factor.replace('.pfm', '.txt')
            with open(depth_factor, 'r') as f:
                ext_l = f.readlines()
            cam_info = ext_l[0].strip()
            cam_info_f = float(cam_info.split(' ')[0].split('[')[1])
            base = float(ext_l[3].strip().split('=')[1])
            doffs = float(ext_l[2].strip().split('=')[1])
            depth_factor = base * cam_info_f
            
            height = 1840
            width = 2300
            
            disp_gt, scale = readPFM(gt_path)
            disp_gt = disp_gt.astype(np.float32)

            disp_gt_copy = disp_gt.copy()
            disp_gt = disp_gt
            invalid_mask = disp_gt == np.inf
            
            depth_gt = depth_factor / (disp_gt + doffs)
            depth_gt = depth_gt / 1000
            depth_gt[invalid_mask] = 0 # set to a invalid number
            disp_gt_copy[invalid_mask] = 0
            edges = get_boundaries(disp_gt_copy, th=1, dilation=0)

            self.gt = depth_gt
            self.edge = edges
        
        elif dataset_name == 'nyu':
            if pred:
                depth_gt = np.load(gt_path.replace('png', 'npy'))
                depth_gt = nn.functional.interpolate(
                    torch.tensor(depth_gt).unsqueeze(dim=0).unsqueeze(dim=0), (480, 640), mode='bilinear', align_corners=True).squeeze().numpy()
                
                edges = get_boundaries(depth_gt, th=1, dilation=0)
            else:
                depth_gt = np.asarray(Image.open(gt_path), dtype=np.float32) / 1000
                edges = get_boundaries(depth_gt, th=1, dilation=0)
            self.gt = depth_gt
            self.edge = edges
            
            
        else:
            raise NotImplementedError
        
        name = name.replace(".npy", "") # u4k
        name = name.replace(".exr", "") # gta
        self.name = name

class ImageDataset:
    def __init__(self, rgb_image_dir, gt_dir=None, dataset_name=''):
        self.rgb_image_dir = rgb_image_dir
        self.files = sorted(os.listdir(self.rgb_image_dir))
        self.gt_dir = gt_dir
        self.dataset_name = dataset_name

        if gt_dir is not None:
            self.gt_dir = gt_dir
            self.gt_files = sorted(os.listdir(self.gt_dir))
            
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.dataset_name == 'nyu':
            return Images(self.rgb_image_dir, self.files, index, self.dataset_name), DepthMap(self.gt_dir, self.gt_files, index, self.dataset_name), DepthMap('/ibex/ai/home/liz0l/codes/ZoeDepth/nfs/save/nyu', self.gt_files, index, self.dataset_name, pred=True)
        if self.gt_dir is not None:
            return Images(self.rgb_image_dir, self.files, index, self.dataset_name), DepthMap(self.gt_dir, self.gt_files, index, self.dataset_name)
        else:
            return Images(self.rgb_image_dir, self.files, index)

def crop(img, crop_bbox):
    crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
    templete = torch.zeros((1, 1, img.shape[-2], img.shape[-1]), dtype=torch.float)
    templete[:, :, crop_y1:crop_y2, crop_x1:crop_x2] = 1.0
    img = img[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
    return img, templete

def generatemask(size):
    # Generates a Guassian mask
    mask = np.zeros(size, dtype=np.float32)
    sigma = int(size[0]/16)
    k_size = int(2 * np.ceil(2 * int(size[0]/16)) + 1)
    mask[int(0.1*size[0]):size[0] - int(0.1*size[0]), int(0.1*size[1]): size[1] - int(0.1*size[1])] = 1
    mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.astype(np.float32)
    return mask

def generatemask_coarse(size):
    # Generates a Guassian mask
    mask = np.zeros(size, dtype=np.float32)
    sigma = int(size[0]/64)
    k_size = int(2 * np.ceil(2 * int(size[0]/64)) + 1)
    mask[int(0.001*size[0]):size[0] - int(0.001*size[0]), int(0.001*size[1]): size[1] - int(0.001*size[1])] = 1
    mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.astype(np.float32)
    return mask

class RunningAverageMap:
    """A dictionary of running averages."""
    def __init__(self, average_map, count_map):
        self.average_map = average_map
        self.count_map = count_map
        self.average_map = self.average_map / self.count_map

    def update(self, pred_map, ct_map):
        self.average_map = (pred_map + self.count_map * self.average_map) / (self.count_map + ct_map)
        self.count_map = self.count_map + ct_map

def regular_tile_param(model, image, offset_x=0, offset_y=0, img_lr=None, iter_pred=None, boundary=0, update=False, avg_depth_map=None, blr_mask=False, crop_size=None,
    img_resolution=None, transform=None):
    # crop size
    # height = 540
    # width = 960
    height = crop_size[0]
    width = crop_size[1]

    assert offset_x >= 0 and offset_y >= 0
    
    tile_num_x = (img_resolution[1] - offset_x) // width
    tile_num_y = (img_resolution[0] - offset_y) // height
    x_start = [width * x + offset_x for x in range(tile_num_x)]
    y_start = [height * y + offset_y for y in range(tile_num_y)]
    imgs_crop = []
    crop_areas = []
    bboxs_roi = []
    bboxs_raw = []

    if iter_pred is not None:
        iter_pred = iter_pred.unsqueeze(dim=0).unsqueeze(dim=0)

    iter_priors = []
    for x in x_start: # w
        for y in y_start: # h
            bbox = (int(y), int(y+height), int(x), int(x+width))
            img_crop, crop_area = crop(image, bbox)
            imgs_crop.append(img_crop)
            crop_areas.append(crop_area)
            crop_y1, crop_y2, crop_x1, crop_x2 = bbox
            bbox_roi = torch.tensor([crop_x1 / img_resolution[1] * 512, crop_y1 / img_resolution[0] * 384, crop_x2 / img_resolution[1] * 512, crop_y2 / img_resolution[0] * 384])
            bboxs_roi.append(bbox_roi)
            bbox_raw = torch.tensor([crop_x1, crop_y1, crop_x2, crop_y2]) 
            bboxs_raw.append(bbox_raw)

            if iter_pred is not None:
                iter_prior, _ = crop(iter_pred, bbox)
                iter_priors.append(iter_prior)

    crop_areas = torch.cat(crop_areas, dim=0)
    imgs_crop = torch.cat(imgs_crop, dim=0)
    bboxs_roi = torch.stack(bboxs_roi, dim=0)
    bboxs_raw = torch.stack(bboxs_raw, dim=0)

    if iter_pred is not None:
        iter_priors = torch.cat(iter_priors, dim=0)
        iter_priors = transform(iter_priors)
        iter_priors = iter_priors.to(image.device).float()

    crop_areas = transform(crop_areas)
    imgs_crop = transform(imgs_crop)

    imgs_crop = imgs_crop.to(image.device).float()
    bboxs_roi = bboxs_roi.to(image.device).float()
    crop_areas = crop_areas.to(image.device).float()
    img_lr = img_lr.to(image.device).float()
    
    pred_depth_crops = []
    with torch.no_grad():
        for i, (img, bbox, crop_area) in enumerate(zip(imgs_crop, bboxs_roi, crop_areas)):

            if iter_pred is not None:
                iter_prior = iter_priors[i].unsqueeze(dim=0)
            else:
                iter_prior = None

            if i == 0:
                out_dict = model(img.unsqueeze(dim=0), mode='eval', image_raw=img_lr, bbox=bbox.unsqueeze(dim=0), crop_area=crop_area.unsqueeze(dim=0), iter_prior=iter_prior if update is True else None)
                whole_depth_pred = out_dict['coarse_depth_pred']
                # return whole_depth_pred.squeeze()
                # pred_depth_crop = out_dict['fine_depth_pred']
                pred_depth_crop = out_dict['metric_depth']
                previous_info = out_dict['previous_info']
            else:
                pred_depth_crop = model(img.unsqueeze(dim=0), mode='eval', image_raw=img_lr, bbox=bbox.unsqueeze(dim=0), crop_area=crop_area.unsqueeze(dim=0), iter_prior=iter_prior if update is True else None, previous_info=previous_info)['metric_depth']
                # pred_depth_crop = model(img.unsqueeze(dim=0), mode='eval', image_raw=img_lr, bbox=bbox.unsqueeze(dim=0), crop_area=crop_area.unsqueeze(dim=0), iter_prior=iter_prior if update is True else None)['fine_depth_pred']


            pred_depth_crop = nn.functional.interpolate(
                pred_depth_crop, (height, width), mode='bilinear', align_corners=True)
            # pred_depth_crop = nn.functional.interpolate(
            #     pred_depth_crop, (height, width), mode='nearest')
            pred_depth_crops.append(pred_depth_crop.squeeze())

    whole_depth_pred = whole_depth_pred.squeeze()
    whole_depth_pred = nn.functional.interpolate(whole_depth_pred.unsqueeze(dim=0).unsqueeze(dim=0), img_resolution, mode='bilinear', align_corners=True).squeeze()

    ####### stich part
    inner_idx = 0
    init_flag = False
    if offset_x == 0 and offset_y == 0:
        init_flag = True
        # pred_depth = whole_depth_pred
        pred_depth = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
    else:
        iter_pred = iter_pred.squeeze()
        pred_depth = iter_pred

    blur_mask = generatemask((height, width)) + 1e-3
    count_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
    blur_mask = torch.tensor(blur_mask, device=whole_depth_pred.device)

    for ii, x in enumerate(x_start):
        for jj, y in enumerate(y_start):
            if init_flag:
                # pred_depth[y: y+height, x: x+width] = blur_mask * pred_depth_crops[inner_idx] + (1 - blur_mask) * crop_temp
                # pred_depth[y: y+height, x: x+width] = blur_mask * pred_depth_crops[inner_idx] + (1 - blur_mask) * crop_temp
                count_map[y: y+height, x: x+width] = blur_mask
                pred_depth[y: y+height, x: x+width] = pred_depth_crops[inner_idx] * blur_mask

            else:
                # ensemble with running mean
                if blr_mask:
                    count_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                    count_map[y: y+height, x: x+width] = blur_mask
                    pred_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                    pred_map[y: y+height, x: x+width] = pred_depth_crops[inner_idx] * blur_mask
                    avg_depth_map.update(pred_map, count_map)
                else:
                    if boundary != 0:
                        count_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                        count_map[y+boundary: y+height-boundary, x+boundary: x+width-boundary] = 1
                        pred_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                        pred_map[y+boundary: y+height-boundary, x+boundary: x+width-boundary] = pred_depth_crops[inner_idx][boundary:-boundary, boundary:-boundary] 
                        avg_depth_map.update(pred_map, count_map)
                    else:
                        count_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                        count_map[y: y+height, x: x+width] = 1
                        pred_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                        pred_map[y: y+height, x: x+width] = pred_depth_crops[inner_idx]
                        avg_depth_map.update(pred_map, count_map)


            inner_idx += 1

    if init_flag:
        avg_depth_map = RunningAverageMap(pred_depth, count_map)
        # blur_mask = generatemask_coarse(img_resolution)
        # blur_mask = torch.tensor(blur_mask, device=whole_depth_pred.device)
        # count_map = (1 - blur_mask)
        # pred_map = whole_depth_pred * (1 - blur_mask)
        # avg_depth_map.update(pred_map, count_map)
        return avg_depth_map

def random_tile_param(model, image, img_lr=None, iter_pred=None, boundary=0, update=False, avg_depth_map=None, blr_mask=False, crop_size=None,
    img_resolution=None, transform=None):
    height = crop_size[0]
    width = crop_size[1]
    
    
    x_start = [random.randint(0, img_resolution[1] - width - 1)]
    y_start = [random.randint(0, img_resolution[0] - height - 1)]
    
    imgs_crop = []
    crop_areas = []
    bboxs_roi = []
    bboxs_raw = []

    if iter_pred is not None:
        iter_pred = iter_pred.unsqueeze(dim=0).unsqueeze(dim=0)

    iter_priors = []
    for x in x_start: # w
        for y in y_start: # h
            bbox = (int(y), int(y+height), int(x), int(x+width))
            img_crop, crop_area = crop(image, bbox)
            imgs_crop.append(img_crop)
            crop_areas.append(crop_area)
            crop_y1, crop_y2, crop_x1, crop_x2 = bbox
            bbox_roi = torch.tensor([crop_x1 / img_resolution[1] * 512, crop_y1 / img_resolution[0] * 384, crop_x2 / img_resolution[1] * 512, crop_y2 / img_resolution[0] * 384])
            bboxs_roi.append(bbox_roi)
            bbox_raw = torch.tensor([crop_x1, crop_y1, crop_x2, crop_y2]) 
            bboxs_raw.append(bbox_raw)

            if iter_pred is not None:
                iter_prior, _ = crop(iter_pred, bbox)
                iter_priors.append(iter_prior)

    crop_areas = torch.cat(crop_areas, dim=0)
    imgs_crop = torch.cat(imgs_crop, dim=0)
    bboxs_roi = torch.stack(bboxs_roi, dim=0)
    bboxs_raw = torch.stack(bboxs_raw, dim=0)

    if iter_pred is not None:
        iter_priors = torch.cat(iter_priors, dim=0)
        iter_priors = transform(iter_priors)
        iter_priors = iter_priors.cuda().float()

    crop_areas = transform(crop_areas)
    imgs_crop = transform(imgs_crop)
    
    imgs_crop = imgs_crop.cuda().float()
    bboxs_roi = bboxs_roi.cuda().float()
    crop_areas = crop_areas.cuda().float()
    img_lr = img_lr.cuda().float()
    
    pred_depth_crops = []
    with torch.no_grad():
        for i, (img, bbox, crop_area) in enumerate(zip(imgs_crop, bboxs_roi, crop_areas)):

            if iter_pred is not None:
                iter_prior = iter_priors[i].unsqueeze(dim=0)
            else:
                iter_prior = None

            if i == 0:
                out_dict = model(img.unsqueeze(dim=0), mode='eval', image_raw=img_lr, bbox=bbox.unsqueeze(dim=0), crop_area=crop_area.unsqueeze(dim=0), iter_prior=iter_prior if update is True else None)
                whole_depth_pred = out_dict['coarse_depth_pred']
                pred_depth_crop = out_dict['metric_depth']
                # return whole_depth_pred.squeeze()
                previous_info = out_dict['previous_info']
            else:
                pred_depth_crop = model(img.unsqueeze(dim=0), mode='eval', image_raw=img_lr, bbox=bbox.unsqueeze(dim=0), crop_area=crop_area.unsqueeze(dim=0), iter_prior=iter_prior if update is True else None, previous_info=previous_info)['metric_depth']


            pred_depth_crop = nn.functional.interpolate(
                pred_depth_crop, (height, width), mode='bilinear', align_corners=True)
            # pred_depth_crop = nn.functional.interpolate(
            #     pred_depth_crop, (height, width), mode='nearest')
            pred_depth_crops.append(pred_depth_crop.squeeze())

    whole_depth_pred = whole_depth_pred.squeeze()

    ####### stich part
    inner_idx = 0
    init_flag = False
    iter_pred = iter_pred.squeeze()
    pred_depth = iter_pred

    blur_mask = generatemask((height, width)) + 1e-3
    blur_mask = torch.tensor(blur_mask, device=whole_depth_pred.device)
    for ii, x in enumerate(x_start):
        for jj, y in enumerate(y_start):
            if init_flag:
                # wont be here
                crop_temp = copy.deepcopy(whole_depth_pred[y: y+height, x: x+width])
                blur_mask = torch.ones((height, width))
                pred_depth[y: y+height, x: x+width] = blur_mask * pred_depth_crops[inner_idx]+ (1 - blur_mask) * crop_temp
            else:

                if blr_mask:
                    count_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                    count_map[y: y+height, x: x+width] = blur_mask
                    pred_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                    pred_map[y: y+height, x: x+width] = pred_depth_crops[inner_idx] * blur_mask
                    avg_depth_map.update(pred_map, count_map)
                else:
                    # ensemble with running mean
                    if boundary != 0:
                        count_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                        count_map[y+boundary: y+height-boundary, x+boundary: x+width-boundary] = 1
                        pred_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                        pred_map[y+boundary: y+height-boundary, x+boundary: x+width-boundary] = pred_depth_crops[inner_idx][boundary:-boundary, boundary:-boundary] 
                        avg_depth_map.update(pred_map, count_map)
                    else:
                        count_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                        count_map[y: y+height, x: x+width] = 1
                        pred_map = torch.zeros(img_resolution, device=pred_depth_crops[inner_idx].device)
                        pred_map[y: y+height, x: x+width] = pred_depth_crops[inner_idx]
                        avg_depth_map.update(pred_map, count_map)

            inner_idx += 1

    if avg_depth_map is None:
        return pred_depth
