# models/discriminator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        def C(in_c, out_c, stride):
            return nn.Sequential(
                spectral_norm(nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1, bias=False)),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            C(in_channels * 2, 64, 2),
            C(64, 128, 2),
            C(128, 256, 2),
            C(256, 512, 1),
            # 最后一层也加谱归一化
            spectral_norm(nn.Conv2d(512, 1, 4, padding=1, bias=False))
        )

    def forward(self, real_input, real_output):
        # 将原图和增强图拼通道
        x = torch.cat([real_input, real_output], dim=1)
        return self.model(x)  # [B,1,H',W']
