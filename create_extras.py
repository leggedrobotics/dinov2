from dinov2.data.datasets import ImageNet

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root="/datasets/ImageNet_FullSize/240712/061417/", extra="/checkpoint/amaia/video/amirbar/processed_datasets/in1k_extras")
    dataset.dump_extra()