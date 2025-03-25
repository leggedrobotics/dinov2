import webdataset as wds
import io
import numpy as np
import json
from PIL import Image
from typing import Callable, Optional
from torchvision.datasets.vision import VisionDataset
import glob
from scipy.interpolate import griddata

class WebDatasetVision(VisionDataset):
    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        images_per_shard=1000,
        shard_pattern: str = "*.tar",  # Pattern for WebDataset shards
        shuffle_buffer: int = 10000,  # Number of samples for shuffle
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.root = root

        self.shard_files = glob.glob(f"{root}/{shard_pattern}")
        self.num_shards = len(self.shard_files)
        self.estimated_num_samples = self.num_shards * images_per_shard  # Approximate dataset size


        # Create the WebDataset pipeline
        self.dataset = (
            wds.WebDataset(self.shard_files, resampled=True, nodesplitter=wds.split_by_node)
            .shuffle(shuffle_buffer)
            .to_tuple("npz", "json")  # Expect .npz images & .json metadata
            .map(self.process_sample)
        )

    def process_sample(self, sample):
        """Process a single sample (depth image & metadata)."""
        npz_data, json_data = sample
        image = self.decode_npz(npz_data)
        metadata = self.safe_json_decode(json_data)
        target = metadata["class_name"]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    @staticmethod
    def decode_npz(npz_data):
        """ Correctly Load .npz depth image from WebDataset"""
        with io.BytesIO(npz_data) as f:
            npz_file = np.load(f)
            depth_img = npz_file["arr_0"]

            # Normalize depth image (invert for visualization)
            depth_min, depth_max = np.min(depth_img), np.max(depth_img)
            depth_normalized = ((depth_img - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            depth_img = np.stack([depth_normalized] * 3, axis=-1)  # Convert to 3-channel

            return Image.fromarray(depth_img)

    @staticmethod
    def safe_json_decode(json_bytes):
        """ Ensure JSON is properly decoded"""
        try:
            json_str = json_bytes.decode("utf-8", errors="ignore")
            return json.loads(json_str)
        except Exception as e:
            print(f"JSON Decode Error: {e}")
            return {}

    def __iter__(self):
        """Returns an iterable over the dataset."""
        return iter(self.dataset)
    
    def __len__(self):
        """Returns an estimated length of the dataset."""
        return self.estimated_num_samples
    
# class WebDatasetVisionPNG(WebDatasetVision):
#     def __init__(
#         self,
#         root: str,
#         transforms: Optional[Callable] = None,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         images_per_shard=3300,
#         shard_pattern: str = "*.tar",  # Pattern for WebDataset shards
#         shuffle_buffer: int = 10000,  # Number of samples for shuffle
#     ):
#         super().__init__(
#             root, transforms, transform, target_transform, images_per_shard, shard_pattern, shuffle_buffer
#         )

#         # Create the WebDataset pipeline
#         self.dataset = (
#             wds.WebDataset(self.shard_files, resampled=True, nodesplitter=wds.split_by_node)
#             .shuffle(shuffle_buffer)
#             .to_tuple("png", "json")  # Expect .png images & .json metadata
#             .map(self.process_sample)
#         )

#     def process_sample(self, sample):
#         """Process a single sample (depth image & metadata)."""
#         png_data, json_data = sample
        
#         image = self.decode_png(png_data)
#         metadata = self.safe_json_decode(json_data)
#         target = metadata["class_name"]

#         if self.transforms is not None:
#             image, target = self.transforms(image, target)

#         return image, target
    
#     def decode_png(self, png_data):
#         """ Correctly Load .png image from WebDataset"""
#         with io.BytesIO(png_data) as f:
#             img = Image.open(f)

#             # Convert single-channel grayscale to 3-channel
#             if img.mode == "L":  # "L" mode means grayscale (single channel)
#                 img = np.array(img)  # Convert to NumPy
#                 img = np.stack([img] * 3, axis=-1)  # Stack into 3-channel
#                 img = Image.fromarray(img)  # Convert back to PIL Image
            
#             return img


class WebDatasetVisionPNG(WebDatasetVision):
    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        images_per_shard=3200,
        shard_pattern: str = "*.tar",  # Pattern for WebDataset shards
        shuffle_buffer: int = 10000,  # Number of samples for shuffle
    ):
        super().__init__(
            root, transforms, transform, target_transform, images_per_shard, shard_pattern, shuffle_buffer
        )

        # Create the WebDataset pipeline
        self.dataset = (
            wds.WebDataset(self.shard_files, resampled=True, nodesplitter=wds.split_by_node, shardshuffle=True)
            .shuffle(shuffle_buffer)
            .to_tuple("png", "json")  # Expect .png images & .json metadata
            .map(self.process_sample)
        )

    def process_sample(self, sample):
        """Process a single sample (depth image & metadata)."""
        png_data, json_data = sample
        
        image = self.decode_png(png_data)
        metadata = self.safe_json_decode(json_data)
        try:
            target = metadata["class_name"]
        except:
            target = "dummy_text"

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    
    def decode_png(self, png_data):
        """ Correctly Load .png image from WebDataset"""
        with io.BytesIO(png_data) as f:
            img = Image.open(f)
            img_np = np.array(img)

            if img_np.dtype == np.uint8:
                # 8-bit grayscale → stack into 3-channel (These are MDE images)
                if img.mode == "L":
                    # Normalize and invert
                    img_np = img_np.astype(np.float32)
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                    img_np = 255.0 - (img_np * 255.0)
                    img_np = img_np.astype(np.uint8)

                    img_np = np.stack([img_np] * 3, axis=-1)
                    return Image.fromarray(img_np)
                else:
                    raise ValueError(f"Unsupported 8-bit image mode: {img.mode}")
            elif img_np.dtype == np.uint16:
                # These are depth images -> COnvert to metric and set the nans to zero
                img_np = (img_np.astype(np.float32) / 65535.0) * 512.0
                img_np[np.isnan(img_np)] = 0
                # clip the value at 20 meters
                img_np = np.clip(img_np, 0, 20)
                # For now lets just min-max normalize the image
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                img_np = (img_np * 255).astype(np.uint8)
                img_np = np.stack([img_np] * 3, axis=-1)
                return Image.fromarray(img_np)
                
            else:
                raise ValueError(f"Unsupported PNG dtype: {img_np.dtype}")
            

class WebDatasetVisionRange(WebDatasetVision):
    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        images_per_shard=3300,
        shard_pattern: str = "*.tar",
        shuffle_buffer: int = 10000,
    ):
        super().__init__(
            root,
            transforms,
            transform,
            target_transform,
            images_per_shard,
            shard_pattern,
            shuffle_buffer,
        )

        # Override dataset pipeline to read npz and json
        self.dataset = (
            wds.WebDataset(self.shard_files, resampled=True, nodesplitter=wds.split_by_node)
            .shuffle(shuffle_buffer)
            .to_tuple("npz", "json")
            .map(self.process_sample)
        )

    def process_sample(self, sample):
        npz_data, json_data = sample
        image = self.decode_npz(npz_data)  # Returns a NumPy array
        metadata = self.safe_json_decode(json_data)
        try:
            target = metadata["class_name"]
        except:
            target = "dummy_text"

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    @staticmethod
    def decode_npz(npz_data):
        """Load a .npz file and return a (H, W, C) NumPy array with interpolated missing values."""
        with io.BytesIO(npz_data) as f:
            npz_file = np.load(f)
            key = list(npz_file.keys())[0]
            array = npz_file[key].astype(np.float32)  # (H, W, C)

            # Identify missing pixels via range channel (channel 0)
            missing_mask = array[..., 0] < 0  # shape (H, W)
            valid_mask = ~missing_mask
        
            if not np.any(valid_mask):
                print("⚠️ All pixels are missing!")
                return array  # No interpolation possible

            H, W, C = array.shape
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            points_valid = np.stack([yy[valid_mask], xx[valid_mask]], axis=-1)

            # Interpolate each channel separately
            for ch in range(C):
                values_valid = array[..., ch][valid_mask]
                points_missing = np.stack([yy[missing_mask], xx[missing_mask]], axis=-1)

                # Perform interpolation only if there are missing values
                if points_missing.size > 0:
                    interp_values = griddata(
                        points_valid, values_valid, points_missing, method="nearest", fill_value=0.0
                    )
                    array[..., ch][missing_mask] = interp_values

            return array  # Now contains filled values
