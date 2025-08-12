# models/NetWithColor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Upsample
from utils import simple_wb

from models.rsgunet import EdgeEnhancer, SELayer, ResidualDenseBlock
# Lab 颜色转换我们用 skimage.color，但只在 loss 里用 numpy
from skimage import color


class NetWithColor(nn.Module):
    def __init__(self, in_ch=3, out_ch=3):
        super().__init__()
        # —— 主干网络 —— （同你原来的一模一样）
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            ResidualDenseBlock(64), SELayer(64),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2, True),
            ResidualDenseBlock(128), SELayer(128),
        )
        self.bottleneck = nn.Sequential(
            ResidualDenseBlock(128), SELayer(128),
            ResidualDenseBlock(128), SELayer(128),
        )
        self.skip1_proj = nn.Conv2d(64, 32, 1, bias=False)
        self.skip2_proj = nn.Conv2d(128, 32, 1, bias=False)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(160, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            ResidualDenseBlock(64),
            SELayer(64),
        )
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            ResidualDenseBlock(32),
            SELayer(32),
        )
        self.final_rgb = nn.Conv2d(32, out_ch, 3, 1, 1)
        nn.init.kaiming_normal_(self.final_rgb.weight, nonlinearity='relu')
        self.edge_enh = EdgeEnhancer()

        # —— 色彩分支 —— 直接拿 enhanced RGB 预测 Lab 的 ab 通道
        self.color_branch = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(16, 2, 1)  # 输出 a,b
        )

    def forward(self, x):
        # —— 主干 ——
        e1 = self.encoder1(x)
        # print("e1:", e1.shape)
        e2 = self.encoder2(e1)
        # print("e2:", e2.shape)
        b  = self.bottleneck(e2)
        # print("b:", b.shape)
        skip2 = self.skip2_proj(e2)
        # print("skip2:", skip2.shape)
        d1_in = torch.cat([b, skip2], dim=1)  # [B,160,64,64]
        d1_up = F.interpolate(d1_in,
                              size=(e1.shape[2], e1.shape[3]),  # 128×128
                              mode='bilinear',
                              align_corners=False)
        d1 = self.dec1_conv(d1_up)  # [B,64,128,128]

        skip1 = self.skip1_proj(e1)  # [B,32,128,128]
        d2_in = torch.cat([d1, skip1], dim=1)  # [B,96,128,128]
        d2_up = F.interpolate(d2_in,
                              size=(x.shape[2], x.shape[3]),  # 256×256
                              mode='bilinear',
                              align_corners=False)
        d2 = self.dec2_conv(d2_up)  # [B,32,256,256]

        # —— 剩余流程：final_rgb、edge_enhancer、color_branch……
        delta = torch.tanh(self.final_rgb(d2)) * 0.1
        rgb = x + delta
        enhanced = self.edge_enh(rgb)
        enhanced = simple_wb(enhanced)
        e01 = (enhanced + 1) * 0.5
        ab_pred = self.color_branch(e01)

        return enhanced, ab_pred

    def color_loss(self, ab_pred, tgt_rgb01):
        """
        ab_pred:        Tensor[B,2,H,W]
        tgt_rgb01:      Tensor[B,3,H,W], 范围 [0,1]
        返回对 ab 通道的 L1 loss
        """
        # 把 tgt_rgb01 拉到 numpy，转换到 Lab
        np_t = (tgt_rgb01.detach().cpu().permute(0,2,3,1).numpy() * 255).astype('uint8')
        lab_t = color.rgb2lab(np_t)       # (B,H,W,3)
        ab_t = torch.from_numpy(lab_t[...,1:])  # (B,H,W,2)
        ab_t = ab_t.permute(0,3,1,2).to(ab_pred.device).float()
        return F.l1_loss(ab_pred, ab_t)


if __name__ == '__main__':
    # 简单 smoke test
    m = NetWithColor().cuda()
    x = torch.randn(2,3,256,256).cuda()
    enhanced, abp = m(x)
    print(enhanced.shape, abp.shape)
    # 预期: torch.Size([2,3,256,256]) torch.Size([2,2,256,256])
