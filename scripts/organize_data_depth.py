import os
import shutil
import tarfile
from glob import glob

# Paths to the dataset files
root_tgt_dir = f"{os.environ['TMPDIR']}/imagenet-1k/"
root_src_dir = "/cluster/scratch/patelm/imagenet-1k-depth"


train_tar_files = [
    "train_images_0.tar.gz",
    "train_images_1.tar.gz",
    "train_images_2.tar.gz",
    "train_images_3.tar.gz",
    "train_images_4.tar.gz",
]
val_tar_file = os.path.join(root_src_dir, "val_images.tar.gz")
test_tar_file = os.path.join(root_src_dir, "test_images.tar.gz")

train_dir = os.path.join(root_tgt_dir, "train")
val_dir = os.path.join(root_tgt_dir, "val")
test_dir = os.path.join(root_tgt_dir, "test")

# Create directories for train, val, test
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

label_path = os.path.join(root_src_dir, "labels.txt")
shutil.copy(label_path, root_tgt_dir)

# Organize test data and rename test images sequentially
with tarfile.open(test_tar_file, "r:gz") as tar:
    tar.extractall(path=test_dir) 

# TODO: Move one level up the test images ?
for root, dirs, files in os.walk(test_dir):
    for file in files:
        # Get the full file path
        file_path = os.path.join(root, file)
        new_filename = file.split('.')[0] + ".JPEG"
        # Move the file to the root of the `train_dir`
        shutil.move(file_path, os.path.join(test_dir, new_filename))

print(f"Test Dataset organized successfully under {test_dir}")

# Extract train tar files into train/<class_name>
for train_tar in train_tar_files:
    train_tar = os.path.join(root_src_dir, train_tar)
    with tarfile.open(train_tar, "r:gz") as tar:
        tar.extractall(path=train_dir)

# Move train files into directories based on their class names
train_images = glob(os.path.join(train_dir, '**', "*.png"), recursive=True)

for image_path in train_images:
    # Extract class name from the filename
    filename = os.path.basename(image_path)
    class_name = filename.split('_')[-1].split('.')[0]  # Class is after the last underscore
    class_dir = os.path.join(train_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    new_filename = filename.rsplit('_', 1)[0] + ".JPEG"  # Keep only the first part
    shutil.move(image_path, os.path.join(class_dir, new_filename))

# Remove now-empty directories
for root, dirs, files in os.walk(train_dir):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        if not os.listdir(dir_path):  # Check if the directory is empty
            os.rmdir(dir_path)

# Organize validation data by class
with tarfile.open(val_tar_file, "r:gz") as tar:
    tar.extractall(path=val_dir)

# Move validation files into directories based on their class names
val_images = glob(os.path.join(val_dir, '**', "*.png"), recursive=True)

for image_path in val_images:
    # Extract class name from the filename
    filename = os.path.basename(image_path)
    class_name = filename.split('_')[-1].split('.')[0]  # Class is after the last underscore
    class_dir = os.path.join(val_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    new_filename = filename.rsplit('_', 1)[0] + ".JPEG"  # Keep only the first part
    shutil.move(image_path, os.path.join(class_dir, new_filename))

# Remove now-empty directories
for root, dirs, files in os.walk(val_dir):
    for dir in dirs:
        dir_path = os.path.join(root, dir)
        if not os.listdir(dir_path):  # Check if the directory is empty
            os.rmdir(dir_path)

print(f"Val Dataset organized successfully under {val_dir}")


