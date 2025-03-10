# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)

# Import DepthAnythingV2
from depth_anything_v2.dpt import DepthAnythingV2

logger = logging.getLogger("dinov2")

class MinMaxNormalize:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val + 1e-8)  # Adding a small epsilon for numerical stability

class LogNormalize:
    def __init__(self, min_depth=0.1, max_depth=100):
        self.log_min = torch.log(torch.tensor(min_depth))
        self.log_max = torch.log(torch.tensor(max_depth))
        
    def __call__(self, tensor: torch.Tensor, min_depth=0.1, max_depth=100) -> torch.Tensor:
        tensor = torch.clamp(tensor, min_depth, max_depth)
        log_depth = torch.log(tensor)
        return (log_depth - self.log_min) / (self.log_max - self.log_min + 1e-8) # Adding a small epsilon for numerical stability

class MaxNormalize:
    def __init__(self, min_depth=0.1, max_depth=100):
        self.min = torch.tensor(min_depth)
        self.max = torch.tensor(max_depth)
        
    def __call__(self, tensor: torch.Tensor, min_depth=0.1, max_depth=100) -> torch.Tensor:
        tensor = torch.clamp(tensor, min_depth, max_depth)
        return (tensor - self.min) / (self.max - self.min + 1e-8) # Adding a small epsilon for numerical stability

class DepthAnythingPreprocessor:
    def __init__(self, model_config='vitl', depth_size=448):
        """
        Initialize the depth preprocessor with DepthAnythingV2
        
        Args:
            model_config: Model configuration ('vits', 'vitb', 'vitl', 'vitg')
            depth_size: Size to resize images for depth estimation
        """
        self.depth_size = depth_size
        
        # Get device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Setup model configs
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        logger.info(f"Initializing DepthAnythingV2 with {model_config} on {self.device}")
        
        # Initialize DepthAnythingV2 model
        self.depth_model = DepthAnythingV2(**model_configs[model_config])
        self.depth_model.load_state_dict(torch.load(f'dinov2/checkpoints_dav2/depth_anything_v2_{model_config}.pth', map_location='cpu'))
        self.depth_model = self.depth_model.to(self.device).eval()
        
        # RGB to tensor transform
        self.to_tensor = transforms.ToTensor()
        
        # Normalizer
        self.normalizer = MinMaxNormalize()
        
    def __call__(self, image):
        """
        Convert an RGB image to a depth map using DepthAnythingV2
        
        Args:
            image: PIL Image
        
        Returns:
            depth_map: Normalized depth map as a tensor
        """
        # Get original size
        original_w, original_h = image.size
        
        # Convert PIL image to tensor (H,W,C) -> (C,H,W)
        img_tensor = self.to_tensor(image)
        
        # Convert grayscale to RGB if needed
        if img_tensor.shape[0] == 1:
            img_tensor = img_tensor.repeat(3, 1, 1)
        
        # Resize for depth estimation
        img_tensor = F.interpolate(
            img_tensor.unsqueeze(0),
            size=(self.depth_size, self.depth_size),
            mode='bilinear',
            align_corners=True
        ).to(self.device)
        
        # Process through depth model
        with torch.no_grad():
            patch_h, patch_w = img_tensor.shape[-2] // 14, img_tensor.shape[-1] // 14
            
            # Get intermediate features
            features = self.depth_model.pretrained.get_intermediate_layers(
                img_tensor,
                self.depth_model.intermediate_layer_idx[self.depth_model.encoder],
                return_class_token=True
            )
            
            # Get depth prediction
            depth = self.depth_model.depth_head(features, patch_h, patch_w)
            depth = F.relu(depth)
            
            depth = depth[0]
            # Resize back to original dimensions
            depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(1), 
                    (original_h, original_w), 
                    mode="bilinear", 
                    align_corners=True
            )[0, 0]
        
        # Normalize to [0, 1]
        depth_normalized = self.normalizer(depth).cpu()
        
        #Convert to 3 channel tensor
        depth_rgb = torch.stack([depth_normalized]*3, dim=0)
        
        return depth_rgb

class DataAugmentationDINODepth(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        use_depth=True,
        depth_model='vitl',
        depth_size=448,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.use_depth = use_depth

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"use_depth: {use_depth}")
        if use_depth:
            logger.info(f"depth_model: {depth_model}")
            logger.info(f"depth_size: {depth_size}")
        logger.info("###################################")

        # Initialize depth preprocessor if needed
        if use_depth:
            self.depth_preprocessor = DepthAnythingPreprocessor(
                model_config=depth_model,
                depth_size=depth_size
            )

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

        # For depth maps, use simpler transformations (just normalize)
        self.depth_normalize = transforms.Compose([make_normalize_transform()])

    def __call__(self, image):
        output = {}

        # Convert RGB to depth if needed
        if self.use_depth:
            # Process image through DepthAnythingV2 to get depth map
            depth_map = self.depth_preprocessor(image)
            # Use depth map instead of RGB for further processing
            # convert to PIl image
            source_image = transforms.ToPILImage()(depth_map)
        else:
            source_image = image
        


        # global crops:
        im1_base = self.geometric_augmentation_global(source_image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(source_image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(source_image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
