from dinov2.data.datasets import ImageNetDepth
import os

for split in ImageNetDepth.Split:
    dataset = ImageNetDepth(split=split, root=f"{os.environ['TMPDIR']}/imagenet-1k", extra=f"{os.environ['TMPDIR']}/imagenet-1k")
    dataset.dump_extra()

# for split in ImageNetDepth.Split:
#     dataset = ImageNetDepth(split=split, root=f"/media/patelm/ssd/imagenet-1k-dav2", extra=f"/media/patelm/ssd/imagenet-1k-dav2")
#     dataset.dump_extra()
