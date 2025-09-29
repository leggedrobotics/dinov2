import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import regnet_y_400mf, regnet_y_800mf, regnet_y_1_6gf, regnet_y_3_2gf, regnet_y_8gf
from torchvision.models import regnet_x_400mf, regnet_x_800mf, regnet_x_1_6gf, regnet_x_3_2gf, regnet_x_8gf
from torchvision.models.feature_extraction import create_feature_extractor

from .bifpn import BiFPN


class RegNetBiFPN(nn.Module):
    def __init__(self, backbone_name="regnet_y_400mf", out_channels=128, n_blocks=1, pretrained=True):
        super().__init__()

        # pick backbone
        backbone_fn = {
            "regnet_x_400mf": regnet_x_400mf,
            "regnet_y_400mf": regnet_y_400mf,
            "regnet_x_800mf": regnet_x_800mf,
            "regnet_y_800mf": regnet_y_800mf,
            "regnet_y_1_6gf": regnet_y_1_6gf,
            "regnet_x_1_6gf": regnet_x_1_6gf,
            "regnet_y_3_2gf": regnet_y_3_2gf,
            "regnet_x_3_2gf": regnet_x_3_2gf,
            "regnet_x_8gf": regnet_x_8gf,
            "regnet_y_8gf": regnet_y_8gf,
        }[backbone_name]

        backbone = backbone_fn(weights="IMAGENET1K_V1" if pretrained else None)

        # extract last 3 stages (C3=/8, C4=/16, C5=/32)
        return_nodes = {
            "trunk_output.block2": "c3",  # /8
            "trunk_output.block3": "c4",  # /16
            "trunk_output.block4": "c5",  # /32 (deepest stage)
        }
        self.backbone = create_feature_extractor(backbone, return_nodes=return_nodes)

        # infer channels from backbone
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            feats = self.backbone(dummy)
        in_channels_list = [f.shape[1] for f in feats.values()]
        print(f"Backbone {backbone_name} feature channels: {in_channels_list}")

        # BiFPN
        self.fpn = BiFPN(in_channels_list, out_channels, n_blocks=n_blocks)

        # pooling layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        # backbone outputs: raw C3, C4, C5
        feats = self.backbone(x)
        feats = OrderedDict([(k, v) for k, v in feats.items()])

        # global feature BEFORE BiFPN (backbone C5 pooled)
        global_feat_backbone = self.global_pool(feats["c5"]).flatten(1)

        # BiFPN fusion → P3, P4, P5
        feats_bifpn = self.fpn(feats)

        # dense features from BiFPN
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

    backbones_x = [
        "regnet_x_400mf", "regnet_x_800mf", "regnet_x_1_6gf", "regnet_x_3_2gf", "regnet_x_8gf"
    ]

    backbones_y = [
        "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_y_3_2gf", "regnet_y_8gf"
    ]

    out_channels = [64, 128, 256, 256, 256]

    for i, backbone in enumerate(backbones_y):
        print(f"\nTesting backbone: {backbone}")
        model = RegNetBiFPN(backbone_name=backbone, out_channels=out_channels[i], n_blocks=1)
        x = torch.randn(1, 3, 40, 64)
        outputs = model(x)
        cls = outputs["global_backbone"]
        dense_feats = outputs["dense_bifpn"]
        print("CLS token:", cls.shape)
        for k, v in dense_feats.items():
            print(k, v.shape)

        print(f"Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        print(f"Backbone params: {sum(p.numel() for p in model.backbone.parameters())/1e6:.2f}M")
        print(f"BiFPN params: {sum(p.numel() for p in model.fpn.parameters())/1e6:.2f}M")

    for i, backbone in enumerate(backbones_x):
        print(f"\nTesting backbone: {backbone}")
        model = RegNetBiFPN(backbone_name=backbone, out_channels=out_channels[i], n_blocks=1)
        x = torch.randn(1, 3, 256, 256)
        outputs = model(x)
        cls = outputs["global_backbone"]
        dense_feats = outputs["dense_bifpn"]
        print("CLS token:", cls.shape)
        for k, v in dense_feats.items():
            print(k, v.shape)

        print(f"Total params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        print(f"Backbone params: {sum(p.numel() for p in model.backbone.parameters())/1e6:.2f}M")
        print(f"BiFPN params: {sum(p.numel() for p in model.fpn.parameters())/1e6:.2f}M")
