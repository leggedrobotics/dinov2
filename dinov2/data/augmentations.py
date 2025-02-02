# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from torchvision import transforms

from .transforms import (
    DEPTH_DEFAULT_MEAN,
    DEPTH_DEFAULT_STD,
    GaussianBlur,
    make_normalize_transform,
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
)


logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        norm='imagenet',
        color_jitter=True,
        gaussian_blur=True,
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

        global_transfo1 = []
        global_transfo2 = []
        local_transfo = []

        # color distorsions / blurring
        if color_jitter:
            print("Using color jitter...")
            color_jittering = transforms.Compose(
                [
                    transforms.RandomApply(
                        [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                        p=0.8,
                    ),
                    transforms.RandomGrayscale(p=0.2),
                ]
            )
            for lst in [global_transfo1, global_transfo2, local_transfo]:
                lst.append(color_jittering)

        if gaussian_blur:
            print("Using gaussian blur...")
            global_transfo1.append(GaussianBlur(p=1.0))
            global_transfo2.append(transforms.Compose(
                [
                    GaussianBlur(p=0.1),
                    transforms.RandomSolarize(threshold=128, p=0.2),
                ]
            ))
            local_transfo.append(GaussianBlur(p=0.5))


        # normalization
        if norm == 'imagenet':
            print("Using imagenet norm...")
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        elif norm == 'minmax':
            print("Using minmax norm...")
            mean = DEPTH_DEFAULT_MEAN
            std = DEPTH_DEFAULT_STD
        else:
            raise ValueError("Wrong norm")
        
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(mean, std),
            ]
        )
        for lst in [global_transfo1, global_transfo2, local_transfo]:
            lst.append(self.normalize)


        self.global_transfo1 = transforms.Compose(global_transfo1)
        self.global_transfo2 = transforms.Compose(global_transfo2)
        self.local_transfo = transforms.Compose(local_transfo)

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
