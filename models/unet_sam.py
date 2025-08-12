# models/unet_sam.py
import torch.nn as nn
from models.rsgunet import RSGUNet

class UNetSAM(RSGUNet):
    def __init__(self, in_ch=4, out_ch=3):
        super().__init__(in_ch=3, out_ch=out_ch)
        # 重建第一个卷积，接受 4 通道输入
        self.enc1[0] = nn.Conv2d(in_ch, 64, 4, 2, 1)  # 原来 in_ch=3 改为 4
