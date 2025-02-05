import os
import h5py
import json
import numpy as np
import tarfile
import io
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Configuration
ROOT_DIR = "/media/sharedaccess/Manthan_RSL_SSD/Indoor_datasets"  # Dataset root
METADATA_FILE = "dataset_info_all.json"
OUTPUT_DIR = "/media/sharedaccess/Manthan_RSL_SSD/Indoor_datasets/indoor_webd"  # WebDataset output path
SHARD_SIZE = 1000  # Number of samples per shard
NUM_WORKERS = min(cpu_count(), 8)  # Use up to 8 CPU cores

# Load metadata file
metadata_path = os.path.join(ROOT_DIR, METADATA_FILE)
with open(metadata_path, "r") as f:
    dataset_info = json.load(f)

# Collect all image indices
image_indices = []
for file_index, entry in enumerate(dataset_info):
    filename = entry["filename"]
    class_name = entry["class_name"]
    image_keys = entry["image_keys"]
    for image_key in image_keys:
        image_indices.append((filename, class_name, image_key))

# Shuffle data to ensure good distribution across shards
np.random.shuffle(image_indices)

# Compute the number of shards
num_shards = len(image_indices) // SHARD_SIZE + int(len(image_indices) % SHARD_SIZE > 0)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Creating {num_shards} shards with multiprocessing using {NUM_WORKERS} workers...")


def process_shard(shard_id):
    """Function to process a single shard."""
    shard_path = os.path.join(OUTPUT_DIR, f"hdf5-webdataset-{shard_id:05d}.tar")
    if os.path.exists(shard_path):  # Skip already processed shards
        print(f"Skipping existing shard: {shard_path}")
        return

    with tarfile.open(shard_path, "w") as tar:
        for idx in range(SHARD_SIZE):
            global_idx = shard_id * SHARD_SIZE + idx
            if global_idx >= len(image_indices):
                break
            
            filename, class_name, image_key = image_indices[global_idx]
            h5_path = os.path.join(ROOT_DIR, filename)

            # Open HDF5 file in read mode (each process opens its own file)
            with h5py.File(h5_path, "r", swmr=True, libver="latest") as h5_file:
                try:
                    # Extract image
                    image_data = h5_file[class_name][image_key][:]

                    # Normalize and process image
                    missing_value = 65535
                    image_data = image_data.astype(np.float32)
                    image_data[image_data == missing_value] = np.nan
                    image_data = 255 * (image_data - np.nanmin(image_data)) / (np.nanmax(image_data) - np.nanmin(image_data))
                    image_data = np.nan_to_num(image_data, nan=0).astype(np.uint8)

                    # Save image as .npz in memory
                    npz_buffer = io.BytesIO()
                    np.savez_compressed(npz_buffer, image_data)
                    npz_bytes = npz_buffer.getvalue()

                    # Write .npz to tar
                    npz_name = f"{class_name}_{image_key}.npz"
                    npz_info = tarfile.TarInfo(npz_name)
                    npz_info.size = len(npz_bytes)
                    tar.addfile(npz_info, io.BytesIO(npz_bytes))

                    # Save metadata as .json
                    metadata = {"class_name": class_name}
                    json_bytes = json.dumps(metadata).encode("utf-8")

                    json_name = f"{class_name}_{image_key}.json"
                    json_info = tarfile.TarInfo(json_name)
                    json_info.size = len(json_bytes)
                    tar.addfile(json_info, io.BytesIO(json_bytes))

                except Exception as e:
                    print(f"❌ Error processing {filename}/{class_name}/{image_key}: {e}")
                    continue

    print(f"✅ Finished processing shard {shard_id}: {shard_path}")


# Run multiprocessing
with Pool(NUM_WORKERS) as pool:
    list(tqdm(pool.imap_unordered(process_shard, range(num_shards)), total=num_shards))

print(f"✅ WebDataset conversion completed successfully! WebDataset stored in {OUTPUT_DIR}")
