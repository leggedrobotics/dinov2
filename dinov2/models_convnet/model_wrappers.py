import torch
import torch.nn as nn
from functools import partial
from dinov2.configs import load_and_merge_config
from dinov2.models import build_model_from_cfg
from dinov2.layers import DINOHead

from dinov2.models_convnet import ResNetBiFPN, RegNetBiFPN, EfficientNetBiFPN

# Teacher wrapper
class DinoTeacher(nn.Module):
    def __init__(self, config_path, ckpt_path, do_ibot=True, do_dino=True):
        super().__init__()
        cfg = load_and_merge_config(config_path)
        backbone, embed_dim = build_model_from_cfg(cfg, only_teacher=True)

        self.ibot_separate_head = cfg.ibot.separate_head
        self.do_ibot = do_ibot
        self.do_dino = do_dino

        dino_head = partial(
            DINOHead,
            in_dim=embed_dim,
            out_dim=cfg.dino.head_n_prototypes,
            hidden_dim=cfg.dino.head_hidden_dim,
            bottleneck_dim=cfg.dino.head_bottleneck_dim,
            nlayers=cfg.dino.head_nlayers,
        )
        self.teacher = nn.ModuleDict({
            "backbone": backbone,
            "dino_head": dino_head()
        })

        # Check if separate iBOT head
        if do_ibot and self.ibot_separate_head:
            ibot_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.ibot.head_n_prototypes,
                hidden_dim=cfg.ibot.head_hidden_dim,
                bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                nlayers=cfg.ibot.head_nlayers,
            )
            self.teacher["ibot_head"] = ibot_head()

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["teacher"]
        self.teacher["backbone"].load_state_dict({
            k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")
        }, strict=True)
        self.teacher["dino_head"].load_state_dict({
            k.replace("dino_head.", ""): v for k, v in state_dict.items() if k.startswith("dino_head.")
        }, strict=True)
        if do_ibot and self.ibot_separate_head:
            self.teacher["ibot_head"].load_state_dict({
                k.replace("ibot_head.", ""): v for k, v in state_dict.items() if k.startswith("ibot_head.")
            }, strict=True)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            feats = self.teacher["backbone"].forward_features(x)

            cls_token = feats["x_norm_clstoken"]
            spatial_tokens = feats["x_norm_patchtokens"]

            # print(cls_token.shape, spatial_tokens.shape)

            if self.do_dino:
                cls_token = self.teacher["dino_head"](cls_token)
            if self.do_ibot:
                if self.ibot_separate_head:
                    spatial_tokens = self.teacher["ibot_head"](spatial_tokens)
                else:
                    spatial_tokens = self.teacher["dino_head"](spatial_tokens)

            return {"cls_token": cls_token, "spatial_tokens": spatial_tokens}



# Student wrapper
def build_student_backbone(backbone_name, out_channels=128):

    if "resnet" in backbone_name:
        student_backbone = ResNetBiFPN(backbone_name=backbone_name, out_channels=out_channels)
    elif "regnet" in backbone_name:
        student_backbone = RegNetBiFPN(backbone_name=backbone_name, out_channels=out_channels)
    elif "efficientnet" in backbone_name:
        student_backbone = EfficientNetBiFPN(backbone_name=backbone_name, out_channels=out_channels)
    else:
        raise ValueError(f"Unknown student type {backbone_name}")

    # Do a dummy forward to initialize
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        outputs = student_backbone(dummy)
        cls = outputs["global_backbone"]
        dense_feats = outputs["dense_bifpn"]

    return student_backbone, cls.shape[1]


class StudentWrapper(nn.Module):
    def __init__(self, backbone_name, head_hidden_dim, head_bottleneck_dim, head_nlayers, teacher_cfg_path, out_channels=128, do_ibot=True, do_dino=True):
        super().__init__()
        backbone, cls_channels = build_student_backbone(backbone_name, out_channels=out_channels)
        cfg = load_and_merge_config(teacher_cfg_path)

        self.ibot_separate_head = cfg.ibot.separate_head
        self.do_ibot = do_ibot
        self.do_dino = do_dino

        # Fix the correct out channels (The backbone out channels)
        dino_head = partial(
            DINOHead,
            in_dim=cls_channels,
            out_dim=cfg.dino.head_n_prototypes,
            hidden_dim=head_hidden_dim,
            bottleneck_dim=head_bottleneck_dim,
            nlayers=head_nlayers,
        )

        self.student = nn.ModuleDict({
            "backbone": backbone,
            "dino_head": dino_head()
        })

        
        if do_ibot: # We always have a separate head for student due to different dims
            ibot_head = partial(
                DINOHead,
                in_dim=out_channels,
                out_dim=cfg.ibot.head_n_prototypes,
                hidden_dim=head_hidden_dim,
                bottleneck_dim=head_bottleneck_dim,
                nlayers=head_nlayers,
            )
            self.student["ibot_head"] = ibot_head()



    def forward(self, x):
        out = self.student["backbone"](x)

        cls_token = out["global_backbone"]
        spatial_tokens = out["dense_bifpn"]["P4"]

        spatial_tokens = spatial_tokens.permute(0, 2, 3, 1).flatten(1, 2)  # B, H*W, C

        # print(cls_token.shape, spatial_tokens.shape)

        if self.do_dino:
            cls_token = self.student["dino_head"](cls_token)

        if self.do_ibot:
            spatial_tokens = self.student["ibot_head"](spatial_tokens)

        return {"cls_token": cls_token, "spatial_tokens": spatial_tokens}