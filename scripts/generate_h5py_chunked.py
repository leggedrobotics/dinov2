import os
import numpy as np
import h5py

def create_hdf5_by_class_original_sizes(root_folder, output_hdf5_path):
    # Initialize the HDF5 file
    with h5py.File(output_hdf5_path, 'w') as h5f:
        # Process each class folder
        for class_name in sorted(os.listdir(root_folder)):
            class_path = os.path.join(root_folder, class_name)
            if not os.path.isdir(class_path):
                continue
            
            # Create a group for each class
            class_group = h5f.create_group(class_name)
            
            # Gather all images in this class
            image_files = [f for f in os.listdir(class_path) if f.endswith('.npz')]
            num_images = len(image_files)
            
            # Process and store each image in its original size for this class
            for idx, file_name in enumerate(image_files):
                file_path = os.path.join(class_path, file_name)
                
                # Load the depth image
                with np.load(file_path) as data:
                    depth_image = data['depth']
                
                # Get the image dimensions
                height, width = depth_image.shape
                
                # Save each image as a separate dataset in the class group with chunking
                image_dset_name = f'image_{idx}'
                
                class_group.create_dataset(
                    image_dset_name, 
                    data=depth_image,
                    dtype=np.float16,
                    compression='lzf',
                    chunks=(height, width)  # Set chunk size to image dimensions
                )

                print(f"Processed {class_name} image {idx + 1}/{num_images} (saved as {image_dset_name})")

    print(f"HDF5 file created with original-sized images at {output_hdf5_path}")

# Usage
create_hdf5_by_class_original_sizes('/media/patelm/ssd/imagenet-1k-dav2/train', '/media/patelm/ssd/imagenet-1k-dav2/train_lzf.hdf5')
