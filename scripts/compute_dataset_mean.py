from torchvision import transforms
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_image(img_path):
    """
    Load the image, convert it to RGB, and return its mean and variance.
    """
    img = Image.open(img_path).convert("RGB")
    img = np.array(img).astype(np.float32) / 255.0  # Scale to [0, 1] range
    mean = img.mean(axis=(0, 1))
    var = img.var(axis=(0, 1))
    return mean, var

def compute_mean_std(dataset_path, max_workers=16):
    # Initialize sums for mean and variance calculation
    num_channels = 3  # 3 channels since the depth is stored as RGB
    mean_sum = np.zeros(num_channels)
    var_sum = np.zeros(num_channels)
    pixel_count = 0

    # Collect all image paths
    img_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for img_file in files:
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                img_paths.append(os.path.join(root, img_file))

    # Use ThreadPoolExecutor for parallel processing of images
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the thread pool
        futures = [executor.submit(process_image, img_path) for img_path in img_paths]

        # Process results as they are completed
        for future in tqdm(as_completed(futures), total=len(futures)):
            mean, var = future.result()
            mean_sum += mean
            var_sum += var
            pixel_count += 1

    # Compute final mean and standard deviation across dataset
    mean = mean_sum / pixel_count
    std = np.sqrt(var_sum / pixel_count)

    return mean, std

# Example usage
dataset_path = '/media/patelm/ssd/imagenet-1k-depth'  # Set your dataset path
mean, std = compute_mean_std(dataset_path)

print(f"Mean: {mean}")
print(f"Std: {std}")
