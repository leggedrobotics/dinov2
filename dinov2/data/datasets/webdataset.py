import webdataset as wds
import io
import numpy as np
import json
from PIL import Image
from typing import Callable, Optional
from torchvision.datasets.vision import VisionDataset
import glob

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