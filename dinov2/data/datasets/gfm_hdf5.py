# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Callable, List, Optional, Tuple, Union, Any
import torch
import numpy as np

from .extended import ExtendedVisionDataset
from .decoders import TargetDecoder, ImageDataDecoder
from torchvision.datasets.folder import DatasetFolder, default_loader
from PIL import Image
import h5py

logger = logging.getLogger("dinov2")

class GFMDataset(ExtendedVisionDataset):

    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root = root

        self.h5_files = []
        self.image_indices = []
        
        for filename in sorted(os.listdir(root)):
            if filename.endswith('.h5') or filename.endswith('.hdf5'):
                print(f"Loading HDF5 file: {filename}")
                h5_path = os.path.join(root, filename)
                h5_file = h5py.File(h5_path, 'r')
                self.h5_files.append(h5_file)
                
                # Gather all class names and image indices within the current file
                for class_name in h5_file.keys():
                    print(f"Loading class: {class_name}")
                    class_group = h5_file[class_name]
                    self.image_indices.extend([(len(self.h5_files) - 1, class_name, image_key) 
                                               for image_key in class_group.keys() if image_key.startswith('image_')])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = self.get_image_data(index)
            # image = ImageDataDecoder(image_data).decode()
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e
        target = self.get_target(index)
        target = TargetDecoder(target).decode()

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    
    def get_image_data(self, index: int) -> Tuple[Any, Any]:
        # try:
        # Retrieve file index, class name, and image key from the image indices
        file_idx, class_name, image_key = self.image_indices[index]
        
        # Access the corresponding HDF5 file and class group
        h5_file = self.h5_files[file_idx]
        class_group = h5_file[class_name]
        
        # Load the image data as a NumPy array
        image = class_group[image_key][:]

        # Handle missing values (specific to certain datasets like Taskonomy)
        missing_value = 65535  # Replace with NaN for handling missing data points
        image = image.astype(np.float32)
        image[image == missing_value] = np.nan

        # We Min-Max normalize the image to [0, 255], Need to avoid this later as we lose resolutoin as well as the metric information
        image = 255 * (image - np.nanmin(image)) / (np.nanmax(image) - np.nanmin(image)) 

        image = image.astype(np.uint8)

        np.nan_to_num(image, copy=False, nan=0)

        image = np.stack([image, image, image], axis=-1)
        return Image.fromarray(image)


    def __len__(self):
        return len(self.image_indices)

    def close(self):
        # Close all open HDF5 files
        for h5_file in self.h5_files:
            h5_file.close()
    
    def get_target(self, index: int):
        return None