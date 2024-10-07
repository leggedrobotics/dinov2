from dinov2.data.datasets import ImageNet
import os

for split in ImageNet.Split:
    dataset = ImageNet(split=split, root=f"{os.environ['TMPDIR']}/imagenet-1k", extra=f"{os.environ['TMPDIR']}/imagenet-1k")
    dataset.dump_extra()
