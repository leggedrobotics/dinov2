# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)


logger = logging.getLogger("dinov2")

class DepthAwareGaussianBlur:
    """Gaussian blur that preserves depth structure better than standard blur."""
    
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max
    
    def __call__(self, tensor):
        if torch.rand(1) < self.p:
            # Use smaller blur radius for depth to preserve edges
            radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
            kernel_size = int(2 * int(2 * radius) + 1)  # Ensure odd kernel size
            if kernel_size >= 3:
                sigma = radius.item()
                return transforms.functional.gaussian_blur(tensor, kernel_size, sigma)
        return tensor

class DepthNoise:
    """Add structured noise appropriate for depth data."""
    
    def __init__(self, p=0.3, noise_std=0.02):
        self.p = p
        self.noise_std = noise_std
    
    def __call__(self, tensor):
        if torch.rand(1) < self.p:
            # Add per-channel noise (since your channels have different meanings)
            noise = torch.randn_like(tensor)
            # Different noise levels per channel
            noise[0] *= self.noise_std * 0.8  # Channel 1: metric depth
            noise[1] *= self.noise_std * 1.0  # Channel 2: clipped depth  
            noise[2] *= self.noise_std * 1.2  # Channel 3: per-image norm 
            
            return torch.clamp(tensor + noise, 0, 1)
        return tensor

class DepthPhotometricJitter:
    """Photometric augmentations adapted for depth data."""
    
    def __init__(self, p=0.8, brightness=0.2, contrast=0.2, per_channel=True):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.per_channel = per_channel
    
    def __call__(self, tensor):
        if torch.rand(1) < self.p:
            # Brightness adjustment (gentle for depth)
            if torch.rand(1) < 0.7:
                if self.per_channel:
                    # Different brightness per channel
                    for c in range(tensor.shape[0]):
                        factor = 1.0 + (torch.rand(1) - 0.5) * 2 * self.brightness
                        tensor[c] = torch.clamp(tensor[c] * factor, 0, 1)
                else:
                    factor = 1.0 + (torch.rand(1) - 0.5) * 2 * self.brightness
                    tensor = torch.clamp(tensor * factor, 0, 1)
            
            # Contrast adjustment
            if torch.rand(1) < 0.7:
                if self.per_channel:
                    for c in range(tensor.shape[0]):
                        factor = 1.0 + (torch.rand(1) - 0.5) * 2 * self.contrast
                        mean_val = tensor[c].mean()
                        tensor[c] = torch.clamp((tensor[c] - mean_val) * factor + mean_val, 0, 1)
                else:
                    factor = 1.0 + (torch.rand(1) - 0.5) * 2 * self.contrast
                    mean_val = tensor.mean()
                    tensor = torch.clamp((tensor - mean_val) * factor + mean_val, 0, 1)
        
        return tensor

class DepthChannelDropout:
    """Randomly zero out channels to encourage robustness."""
    
    def __init__(self, p=0.1):
        self.p = p
    
    def __call__(self, tensor):
        if torch.rand(1) < self.p:
            # Never drop all channels, and prefer dropping channel 3 (least critical)
            dropout_probs = [0.05, 0.05, 0.10]  # Lower prob for ch1,ch2, higher for ch3
            for c in range(tensor.shape[0]):
                if torch.rand(1) < dropout_probs[c]:
                    tensor[c] = 0
        return tensor

class DepthNormalize:
    """Normalization for 3-channel depth data."""
    
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
    
    def __call__(self, tensor):
        return (tensor - self.mean) / (self.std + 1e-8)

class RandomPixelDropout:
    """Randomly drop pixels across all channels (simulate depth sensor noise)."""
    def __init__(self, p=0.3, drop_prob_range=(0.0, 0.1)):
        """
        p: probability of applying dropout
        drop_prob_range: range of dropout fraction (e.g., (0.0, 0.1) = up to 10% pixels)
        """
        self.p = p
        self.drop_prob_range = drop_prob_range

    def __call__(self, tensor):
        if torch.rand(1) < self.p:
            # Sample dropout probability for this call
            drop_prob = torch.empty(1).uniform_(*self.drop_prob_range).item()
            
            if drop_prob > 0:
                mask = torch.rand_like(tensor[0]) < drop_prob  # same mask across all channels
                tensor[:, mask] = 0.0
        return tensor

class DataAugmentationDINODepthNorm(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale, 
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):

        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size 
        self.local_crops_size = local_crops_size

        # Hardcoded sizes for High Resolution Fine Tuning
        # self.global_crops_size = 518
        # self.local_crops_size = 224

        logger.info("###################################")
        logger.info("Using DEPTH data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [   
                transforms.ToTensor(),
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [   
                transforms.ToTensor(),
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(p=0.5),   
            ]
        )

        # Depth-specific photometric augmentations (NO ColorJitter, Grayscale, Solarize)
        depth_photometric = DepthPhotometricJitter(
            p=0.8, 
            brightness=0.20,  # More conservative than RGB
            contrast=0.20,    # More conservative than RGB
            per_channel=True
        )

        # Depth-aware blur and noise
        global_extra1 = transforms.Compose([
            DepthAwareGaussianBlur(p=1.0, radius_max=2.0),  # Less blur than RGB
            DepthNoise(p=0.3, noise_std=0.03),           # Slightly more noise than RGB
            RandomPixelDropout(p=0.3, drop_prob_range=(0.0, 0.1)),  # Up to 10% pixels dropped
            # DepthChannelDropout(p=0.1),
        ])

        global_extra2 = transforms.Compose([
            DepthAwareGaussianBlur(p=0.1, radius_max=1.5),
            DepthNoise(p=0.2, noise_std=0.02),
        ])

        local_extra = transforms.Compose([
            DepthAwareGaussianBlur(p=0.5, radius_max=1.5),
            DepthNoise(p=0.25, noise_std=0.018),
            RandomPixelDropout(p=0.2, drop_prob_range=(0.0, 0.05)),  # Up to 10% pixels dropped
        ])

        self.normalize = transforms.Compose(
            [
                make_normalize_transform(),
            ]
        )

        # Complete transformation pipelines
        self.global_transfo1 = transforms.Compose([
            depth_photometric, 
            global_extra1, 
            self.normalize
        ])
        
        self.global_transfo2 = transforms.Compose([
            depth_photometric, 
            global_extra2, 
            self.normalize
        ])
        
        self.local_transfo = transforms.Compose([
            depth_photometric, 
            local_extra, 
            self.normalize
        ])
        # self.global_transfo1 = transforms.Compose([self.normalize])
        # self.global_transfo2 = transforms.Compose([self.normalize])
        # self.local_transfo = transforms.Compose([self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output

class DataAugmentationDINODepthNormConv(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale, 
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):

        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size 
        self.local_crops_size = local_crops_size

        # Hardcoded sizes for High Resolution Fine Tuning
        # self.global_crops_size = 518
        # self.local_crops_size = 224

        logger.info("###################################")
        logger.info("Using DEPTH data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # Get the global crop size for convnet
        global_crop_size_conv = (global_crops_size // 14 ) * 16

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [   
                transforms.ToTensor(),
                transforms.RandomResizedCrop(
                    global_crop_size_conv, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [   
                transforms.ToTensor(),
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.RandomHorizontalFlip(p=0.5),   
            ]
        )

        self.geometric_resize_global = transforms.Compose(
            [
                transforms.Resize(global_crops_size, interpolation=transforms.InterpolationMode.BILINEAR)
            ]
        )

        # Depth-specific photometric augmentations (NO ColorJitter, Grayscale, Solarize)
        depth_photometric = DepthPhotometricJitter(
            p=0.8, 
            brightness=0.20,  # More conservative than RGB
            contrast=0.20,    # More conservative than RGB
            per_channel=True
        )

        # Depth-aware blur and noise
        global_extra1 = transforms.Compose([
            DepthAwareGaussianBlur(p=1.0, radius_max=2.0),  # Less blur than RGB
            DepthNoise(p=0.3, noise_std=0.03),           # Slightly more noise than RGB
            RandomPixelDropout(p=0.3, drop_prob_range=(0.0, 0.1)),  # Up to 10% pixels dropped
            # DepthChannelDropout(p=0.1),
        ])

        global_extra2 = transforms.Compose([
            DepthAwareGaussianBlur(p=0.1, radius_max=1.5),
            DepthNoise(p=0.2, noise_std=0.02),
        ])

        local_extra = transforms.Compose([
            DepthAwareGaussianBlur(p=0.5, radius_max=1.5),
            DepthNoise(p=0.25, noise_std=0.018),
            RandomPixelDropout(p=0.2, drop_prob_range=(0.0, 0.05)),  # Up to 10% pixels dropped
        ])

        self.normalize = transforms.Compose(
            [
                make_normalize_transform(),
            ]
        )

        # Complete transformation pipelines
        self.global_transfo1 = transforms.Compose([
            depth_photometric, 
            global_extra1, 
            self.normalize
        ])
        
        self.global_transfo2 = transforms.Compose([
            depth_photometric, 
            global_extra2, 
            self.normalize
        ])
        
        self.local_transfo = transforms.Compose([
            depth_photometric, 
            local_extra, 
            self.normalize
        ])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        global_crop_teacher_1 = self.geometric_resize_global(global_crop_1)
        global_crop_teacher_2 = self.geometric_resize_global(global_crop_2)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_teacher_1, global_crop_teacher_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output

class DataAugmentationDINODepth(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

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

        # # normalization
        # self.normalize = transforms.Compose(
        #     [
        #         MinMaxNormalize(),
        #         make_normalize_transform(),
        #     ]
        # )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])
        # self.global_transfo1 = transforms.Compose([self.normalize])
        # self.global_transfo2 = transforms.Compose([self.normalize])
        # self.local_transfo = transforms.Compose([self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
