import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2,
    efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,
)
from torchvision.models.feature_extraction import create_feature_extractor

from .bifpn import BiFPN


class EfficientNetBiFPN(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", out_channels=128, n_blocks=1, pretrained=True):
        super().__init__()

        # pick backbone
        backbone_fn = {
            "efficientnet_b0": efficientnet_b0,
            "efficientnet_b1": efficientnet_b1,
            "efficientnet_b2": efficientnet_b2,
            "efficientnet_b3": efficientnet_b3,
            "efficientnet_b4": efficientnet_b4,
            "efficientnet_b5": efficientnet_b5,
            "efficientnet_b6": efficientnet_b6,
            "efficientnet_b7": efficientnet_b7,

        }[backbone_name]

        backbone = backbone_fn(weights="IMAGENET1K_V1" if pretrained else None)

        # extract stages (from torchvision EfficientNet impl)
        return_nodes = {
            "features.3": "c3",  # /8
            "features.4": "c4",  # /16
            "features.6": "c5",  # /32
        }
        self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)

        # infer channels
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            feats = self.backbone(dummy)
        in_channels_list = [f.shape[1] for f in feats.values()]
        print(f"Backbone {backbone_name} feature channels: {in_channels_list}")

        # BiFPN
        self.fpn = BiFPN(in_channels_list, out_channels, n_blocks=n_blocks)

        # pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feats = self.backbone(x)  # dict with c3, c4, c5
        feats = OrderedDict([(k, v) for k, v in feats.items()])

        # Global CLS proxy = pooled backbone C5
        global_feat_backbone = self.global_pool(feats["c5"]).flatten(1)

        # BiFPN
        feats_bifpn = self.fpn(feats)

        # Dense maps
        dense_feats_bifpn = {
            "P3": feats_bifpn["c3"],
            "P4": feats_bifpn["c4"],
            "P5": feats_bifpn["c5"],
        }

        return {
            "global_backbone": global_feat_backbone,
            "dense_bifpn": dense_feats_bifpn,
        }


if __name__ == "__main__":
    backbones = ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
                 "efficientnet_b3", "efficientnet_b4", "efficientnet_b5",
                 "efficientnet_b6", "efficientnet_b7"]
    out_channels = [64, 128, 128, 256, 256, 256, 256, 256]  # recommended BiFPN widths

    for i, backbone in enumerate(backbones):
        print(f"\nTesting backbone: {backbone}")
        model = EfficientNetBiFPN(backbone_name=backbone, out_channels=out_channels[i], n_blocks=1)
        x = torch.randn(1, 3, 256, 256)
        out = model(x)

        print("Global backbone feature:", out["global_backbone"].shape)
        for k, v in out["dense_bifpn"].items():
            print(k, v.shape)

        print(f"Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        print(f"Backbone params: {sum(p.numel() for p in model.backbone.parameters())/1e6:.2f}M")
        print(f"BiFPN params: {sum(p.numel() for p in model.fpn.parameters())/1e6:.2f}M")
