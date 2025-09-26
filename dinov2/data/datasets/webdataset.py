import webdataset as wds
import io
import numpy as np
import json
from PIL import Image
from typing import Callable, Optional, List
from torchvision.datasets.vision import VisionDataset
import glob
from scipy.interpolate import griddata
import os
import cv2

def custom_selector(sample):
    """
    Selects the correct key combination from WebDataset sample.
    Supports both ("png", "json") and ("depth.png", "meta.json").
    Returns a tuple or None to be filtered out.
    """
    if "png" in sample and "json" in sample:
        return sample["png"], sample["json"]
    elif "depth.png" in sample and "meta.json" in sample:
        return sample["depth.png"], sample["meta.json"]
    else:
        return None  # Skip this sample

class WebDatasetVision(VisionDataset):
    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        images_per_shard=1000,
        shard_pattern: str = "*.tar",
        shuffle_buffer: int = 1000,
        dataset_list_file: Optional[str] = None,  # New parameter for dataset selection
        shard_files: Optional[List[str]] = None,  # Allow direct shard file input
    ):
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        
        # Get shard files based on dataset selection
        if shard_files is not None:
            self.shard_files = shard_files
        else:
            self.shard_files = self._get_shard_files(root, shard_pattern, dataset_list_file)

        self.num_shards = len(self.shard_files)
        self.estimated_num_samples = self.num_shards * images_per_shard

        # Create the WebDataset pipeline
        self.dataset = (
            wds.WebDataset(self.shard_files, resampled=True, nodesplitter=wds.split_by_node)
            .shuffle(shuffle_buffer)
            .to_tuple("npz", "json")  # Expect .npz images & .json metadata
            .map(self.process_sample)
        )

    def _get_shard_files(self, root: str, shard_pattern: str, dataset_list_file: Optional[str]) -> List[str]:
        """Get shard files, optionally filtered by dataset list file."""
        
        if dataset_list_file and os.path.exists(dataset_list_file):
            # Read dataset list from file
            with open(dataset_list_file, 'r') as f:
                dataset_folders = [line.strip() for line in f if line.strip()]
            
            shard_files = []
            for folder in dataset_folders:
                folder_path = os.path.join(root, folder)
                if os.path.exists(folder_path):
                    folder_shards = glob.glob(os.path.join(folder_path, shard_pattern))
                    shard_files.extend(folder_shards)
                    print(f"📁 Added {len(folder_shards)} shards from {folder}")
                else:
                    print(f"⚠️ Dataset folder not found: {folder_path}")
            
            return sorted(shard_files)
        else:
            # Original logic for automatic detection
            contains_subfolders = any(os.path.isdir(os.path.join(root, entry)) 
                                    for entry in os.listdir(root))

            if contains_subfolders:
                shard_files = glob.glob(os.path.join(root, "**", shard_pattern), recursive=True)
            else:
                shard_files = glob.glob(os.path.join(root, shard_pattern))
            
            return sorted(shard_files)
    
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


def not_none(x):
    return x is not None

class WebDatasetVisionPNG(WebDatasetVision):
    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        images_per_shard=3200,
        shard_pattern: str = "*.tar",
        shuffle_buffer: int = 1000,
        dataset_list_file: Optional[str] = None,  # New parameter
        shard_files: Optional[List[str]] = None,  # Allow direct shard file input
    ):
        super().__init__(
            root, transforms, transform, target_transform, 
            images_per_shard, shard_pattern, shuffle_buffer, dataset_list_file, shard_files
        )

        self.dataset = (
            wds.WebDataset(self.shard_files, resampled=True, 
                            nodesplitter=wds.split_by_node, shardshuffle=True)
            .shuffle(shuffle_buffer)
            .decode()
            .map(custom_selector)
            .select(not_none)
            .map(self.process_sample)
        )

    def process_sample(self, sample):
        """Process sample from flexible selector."""
        png_data, json_data = sample
        
        metadata = json_data
        try:
            target = metadata["class_name"]
        except:
            # target = "dummy_text"
            target = metadata.get("dataset", "sa-1b")

        # Decode to 2-channel depth image
        image = self.decode_png_three_channel(png_data, metadata)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    
    def decode_png_three_channel(self, png_data, metadata, depth_multiplier=None, max_depth_ch0=100.0, max_depth_ch1=10.0):
        """
        Decode PNG using cv2 and return 2-channel depth image:
        - Channel 1: Metric depth normalized as ln(1 + depth) / ln(101)
        - Channel 2: Per-image min-max normalized as (ln(1 + depth) - ln(1 + min_depth)) / (ln(1 + max_depth) - ln(1 + min_depth))
        """
        try:
            # Decode using cv2
            img_array = np.frombuffer(png_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                raise ValueError("Failed to decode PNG with cv2")
            
            img_np = img.astype(np.float32)


            dataset = metadata.get("dataset", "Unknown")

            # Get depth multiplier from metadata if not provided
            if depth_multiplier is None:
                depth_multiplier = metadata.get("depth_multiplier", 1.0)
                depth_resolution = metadata.get("depth_resolution", 1.0)

                # Special case for somne dataset where the keyword is "depth_resolution" instead of "depth_multiplier"
                if depth_resolution != 1.0:
                    depth_multiplier = depth_resolution
                
                # Special case for MetaGraspNetSyn
                if dataset == "MetaGraspNetSyn":
                    depth_multiplier = depth_multiplier / 100.0

            # Handle different data types
            if img.dtype == np.uint8:
                # This is inverse depth data - need to invert it first
                if len(img_np.shape) == 2:
                    # Normalize to [0, 1]
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                    metric_depth = max_depth_ch1 * np.exp(-5.0 * img_np)  # Exponential decay

                else:
                    raise ValueError(f"Unsupported 8-bit image shape: {img_np.shape}")
            
            elif img.dtype == np.uint16:
                # This has metric depth information
                if dataset == "hm3d" or dataset == "taskonomy":
                    img_np[img_np >= 65530] = 0 # Also change 65535 to 0 since it is not a valid depth
                img_np[np.isnan(img_np)] = 0
                metric_depth = img_np / depth_multiplier
                metric_depth[np.isnan(metric_depth)] = 0
                metric_depth = np.clip(metric_depth, 0, max_depth_ch0)
            
            else:
                raise ValueError(f"Unsupported PNG dtype: {img.dtype}")
            
            # Channel 1: Metric depth normalized (100 is max depth)
            log_depth = np.log1p(metric_depth)  # ln(1 + depth)
            channel_1 = log_depth / np.log1p(max_depth_ch0)  # Normalize by ln(101)

            # Channel 2:  Metric depth normalized (10 is max depth)
            channel_2 = np.clip(log_depth / np.log(max_depth_ch1), 0, 1)  
            
            # Channel 3: Per-image min-max normalized
            min_log_depth = np.log1p(metric_depth.min())
            max_log_depth = np.log1p(metric_depth.max())
            
            # Avoid division by zero
            if max_log_depth > min_log_depth:
                channel_3 = (log_depth - min_log_depth) / (max_log_depth - min_log_depth)
            else:
                channel_3 = np.zeros_like(log_depth)
            
            # Combine channels - shape will be (H, W, 2)
            three_channel_depth = np.stack([channel_1, channel_2, channel_3], axis=-1)
            
            return three_channel_depth  # Return numpy array
            
        except Exception as e:
            print(f"⚠️ PNG decode failed: {e} — Using blank 3-channel image.")
            # Return a blank 3-channel depth image
            return np.zeros((224, 224, 3), dtype=np.float32)


class WebDatasetVisionPNGv1(WebDatasetVision):
    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        images_per_shard=3200,
        shard_pattern: str = "*.tar",  # Pattern for WebDataset shards
        shuffle_buffer: int = 1000,  # Number of samples for shuffle
    ):
        super().__init__(
            root, transforms, transform, target_transform, images_per_shard, shard_pattern, shuffle_buffer
        )

        # # Create the WebDataset pipeline
        # self.dataset = (
        #     wds.WebDataset(self.shard_files, resampled=True, nodesplitter=wds.split_by_node, shardshuffle=True)
        #     .shuffle(shuffle_buffer)
        #     .to_tuple("png", "json")  # Expect .png images & .json metadata
        #     .map(self.process_sample, handler=wds.handlers.ignore_and_continue)
        # )
        self.dataset = (
            wds.WebDataset(self.shard_files, resampled=True, 
                            nodesplitter=wds.split_by_node, shardshuffle=True)
            .shuffle(shuffle_buffer)
            .decode()
            .map(custom_selector)
            .select(not_none)
            .map(self.process_sample)
        )

    def process_sample(self, sample):
        """Process a single sample (depth image & metadata)."""
        png_data, json_data = sample
        
        # metadata = self.safe_json_decode(json_data)
        metadata = json_data
        try:
            target = metadata["class_name"]
        except:
            target = "dummy_text"

        # Check if depth _resolution or depth_multiplier is in metadata
        depth_multiplier = metadata.get("depth_multiplier", 1.0)
        depth_resolution = metadata.get("depth_resolution", 1.0) # This was saved for omnidata 
        
        if depth_resolution != 1.0:
            # If depth_resolution is not 1.0, use it to scale the image
            depth_multiplier = depth_resolution

        image = self.decode_png(png_data, depth_multiplier=depth_multiplier)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
    
    def decode_png(self, png_data, depth_multiplier=1.0, clip_depth=20):
        """ Robustly load .png image from WebDataset """
        try:
            with io.BytesIO(png_data) as f:
                img = Image.open(f)
                img.load()  # Force load to trigger early failure
                img_np = np.array(img)

                if img_np.dtype == np.uint8:
                    if img.mode == "L":
                        img_np = img_np.astype(np.float32)
                        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                        img_np = 255.0 - (img_np * 255.0)
                        img_np = img_np.astype(np.uint8)
                        img_np = np.stack([img_np] * 3, axis=-1)
                        return Image.fromarray(img_np)
                    else:
                        raise ValueError(f"Unsupported 8-bit image mode: {img.mode}")
                elif img_np.dtype == np.uint16:
                    img_np[np.isnan(img_np)] = 0
                    img_np = img_np.astype(np.float32) / depth_multiplier
                    img_np[np.isnan(img_np)] = 0
                    img_np = np.clip(img_np, 0, clip_depth)
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                    img_np = (img_np * 255).astype(np.uint8)
                    img_np = np.stack([img_np] * 3, axis=-1)
                    return Image.fromarray(img_np)
                else:
                    raise ValueError(f"Unsupported PNG dtype: {img_np.dtype}")
        except Exception as e:
            print(f"⚠️ PNG decode failed: {e} — Using blank image.")
            # Return a blank (black) 3-channel image (e.g., 224x224)
            return Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8))

        
class WebDatasetVisionRange(WebDatasetVision):
    def __init__(
        self,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        images_per_shard=3300,
        shard_pattern: str = "*.tar",
        shuffle_buffer: int = 1000,
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