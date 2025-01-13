import os
import json
import h5py

def save_image_keys(root: str, output_file: str) -> None:
    dataset_info = []
    
    # Iterate over all HDF5 files in the root directory
    for filename in sorted(os.listdir(root)):
        if filename.endswith('.h5') or filename.endswith('.hdf5'):
            h5_path = os.path.join(root, filename)
            print(f"Processing HDF5 file: {filename}")
            with h5py.File(h5_path, 'r') as h5_file:
                # Gather class names and image keys within the current file
                for class_name in h5_file.keys():
                    class_group = h5_file[class_name]
                    image_keys = [image_key for image_key in class_group.keys() if image_key.startswith('image_')]
                    # Store the file name, class name, and image keys
                    dataset_info.append({
                        "filename": filename,
                        "class_name": class_name,
                        "image_keys": image_keys
                    })

    # Save the collected information to a JSON file
    with open(output_file, 'w') as f:
        json.dump(dataset_info, f)
    print(f"Saved dataset info to {output_file}")

# Usage
root = "/media/patelm/Manthan_RSL_SSD/Indoor_datasets/imagenet_hdf5"       # Directory containing HDF5 files
output_file = "/media/patelm/Manthan_RSL_SSD/Indoor_datasets/imagenet_hdf5/dataset_info.json"  # Output JSON file
save_image_keys(root, output_file)

