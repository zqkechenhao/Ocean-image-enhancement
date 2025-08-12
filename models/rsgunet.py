# models/rsgunet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Upsample

from utils import simple_wb


class SELayer(nn.Module):
    """增强版通道注意力机制"""
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualDenseBlock(nn.Module):
    """残差密集块"""
    def __init__(self, in_c, growth_rate=32, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, in_c, 3, padding=1)
        self.conv2 = nn.Conv2d(in_c, in_c, 3, padding=1)
        self.conv3 = nn.Conv2d(in_c, in_c, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.se = SELayer(in_c, reduction)
    def forward(self, x):
        identity = x
        out = self.lrelu(self.conv1(x))
        out = self.lrelu(self.conv2(out))
        out = self.conv3(out)
        out = self.se(out)
        return identity + out * 0.2

class EdgeEnhancer(nn.Module):
    """在解码端加入的轮廓强化模块"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=1),
        )
        # Precompute Sobel kernels
        sobel_x = torch.tensor([[[[-1., 0., 1.],
                                  [-2., 0., 2.],
                                  [-1., 0., 1.]]]])
        sobel_y = torch.tensor([[[[-1., -2., -1.],
                                  [ 0.,  0.,  0.],
                                  [ 1.,  2.,  1.]]]])
        # make [3,1,3,3]
        self.register_buffer('sobel_x', sobel_x.repeat(3,1,1,1))
        self.register_buffer('sobel_y', sobel_y.repeat(3,1,1,1))

    def forward(self, x):
        # x assumed RGB ([B,3,H,W])
        gx = F.conv2d(x, self.sobel_x, groups=3, padding=1)
        gy = F.conv2d(x, self.sobel_y, groups=3, padding=1)
        edge = torch.sqrt(gx*gx + gy*gy + 1e-6)
        return x + self.conv(edge)


class RSGUNet(nn.Module):
    """RSGUNet 主干"""
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        self.edge_enhancer = EdgeEnhancer()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualDenseBlock(64),
            SELayer(64),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualDenseBlock(128),
            SELayer(128),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualDenseBlock(128),
            SELayer(128),
            ResidualDenseBlock(128),
            SELayer(128),
        )

        # ---- 降维 1×1 卷积：把 enc1/enc2 的通道从 64→32 ----
        self.skip1_proj = nn.Conv2d(64, 32, 1, bias=False)
        self.skip2_proj = nn.Conv2d(128, 32, 1, bias=False)

        # Decoder

        # dec1_conv: 改为 Upsample + Conv
        self.dec1_conv = nn.Sequential(
            Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64→128
            nn.Conv2d(160, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            ResidualDenseBlock(64),
            SELayer(64),
        )
        # dec2
        self.dec2_conv = nn.Sequential(
            Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 128→256
            nn.Conv2d(96, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            ResidualDenseBlock(32),
            SELayer(32),
        )

        # 最后再映射到 out_ch
        self.final = nn.Conv2d(32, out_ch, 3, 1, 1)
        nn.init.kaiming_normal_(self.final.weight, nonlinearity='relu')
        if self.final.bias is not None:
            nn.init.zeros_(self.final.bias)

    def forward(self, x):
        # ---- Encoder ----
        e1 = self.enc1(x)  # [B,64,128,128]
        e2 = self.enc2(e1)  # [B,128,64,64]
        b = self.bottleneck(e2)  # [B,128,64,64]

        # ---- 降维并上采样 skip2 ----
        skip2 = self.skip2_proj(e2)  # [B,32,64,64]
        # dec1 上采样到 128×128
        d1 = self.dec1_conv(torch.cat([b, skip2], dim=1))

        # ---- 降维 skip1 ----
        skip1 = self.skip1_proj(e1)  # [B,32,128,128]
        # dec2 上采样到 256×256
        d2 = self.dec2_conv(torch.cat([d1, skip1], dim=1))

        # ---- 输出残差 & 边缘增强 ----
        delta = self.final(d2)  # [B,3,256,256]
        delta = torch.tanh(delta) * 0.1
        pred = x[:, :3, :, :] + delta
        enhanced = self.edge_enhancer(pred)
        enhanced = simple_wb(enhanced)
        enhanced = enhanced.clamp(-1.0, 1.0)
        return enhanced

