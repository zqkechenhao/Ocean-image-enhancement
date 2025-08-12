# RSGUnet_proj/VGG/vgg_feature_extractor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG19_Weights
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 将 Torch Hub 缓存目录改为项目下的 vgg_cache 文件夹
torch.hub.set_dir(r"D:\PycharmProjects\imageEnhance\RSGUnet_proj\dataset\vgg_cache")

class VGGPerceptualExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        vgg_feats = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        feats = vgg_feats.features.to(device).eval()

        # 拆分为 6 个 block, 添加 conv5_x
        self.block0 = nn.Sequential(*feats[:2]).to(device).eval()   # conv1_1 + relu1_1
        self.block1 = nn.Sequential(*feats[2:5]).to(device).eval()  # conv1_2 + relu1_2
        self.block2 = nn.Sequential(*feats[5:10]).to(device).eval() # conv2_1~conv2_2 + relu2_2
        self.block3 = nn.Sequential(*feats[10:19]).to(device).eval()# conv3_1~conv3_4 + relu3_4
        self.block4 = nn.Sequential(*feats[19:28]).to(device).eval()# conv4_1~conv4_4 + relu4_4
        self.block5 = nn.Sequential(*feats[28:]).to(device).eval()  # conv5_1~conv5_4 + relu5_4

        for blk in (self.block0, self.block1, self.block2, self.block3, self.block4, self.block5):
            for p in blk.parameters():
                p.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        x = (x + 1) * 0.5
        x = (x - self.mean) / self.std

        f0 = self.block0(x)
        f1 = self.block1(f0)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        f5 = self.block5(f4)
        return [f0, f1, f2, f3, f4, f5]


def compute_vgg_loss(vgg_extractor: VGGPerceptualExtractor,
                     pred: torch.Tensor,
                     target: torch.Tensor,
                     weights=None) -> torch.Tensor:
    """
    多尺度加权感知损失：浅层权重大，深层权重小。
    pred, target: [B,3,H,W] 且归一化至 [0,1]
    """
    if weights is None:
        weights = [1.0, 1.0, 0.8, 0.5, 0.3, 0.1]  # 加入 conv5，减小其权重

    with torch.no_grad():
        target_feats = vgg_extractor(target)
    pred_feats = vgg_extractor(pred)

    loss = 0.0
    for w, pf, tf in zip(weights, pred_feats, target_feats):
        loss += w * F.mse_loss(pf, tf)
    return loss

