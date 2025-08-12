import cv2
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from RSGUnet_proj.dataset.euvp_dataset import EUVPPairedDataset


def _random_hue_shift(img_np):
    """随机色调偏移（模拟珊瑚色彩变化）"""
    hsv = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
    hue_shift = random.randint(-5, 5)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _random_sharpen(img_np):
    """随机锐化增强（珊瑚纹理强化）"""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img_np, -1, kernel)


def _physical_preprocess(img_np):
    """物理预处理流水线"""
    img_np = enhance_color_correction(img_np)
    img_np = enhance_brightness(img_np)
    img_np = enhance_sharpness(img_np)
    return img_np


class OceanPairedDataset(Dataset):
    """海洋图像增强数据集（针对珊瑚特征优化）"""

    def __init__(self, root_dir, split='train', transform=None):
        """
        初始化数据集
        :param root_dir: 数据集根目录
        :param split: 数据集划分 (train/val/test)
        :param transform: 图像变换组合
        """
        self.base = EUVPPairedDataset(root_dir, split=split, transform=None)
        self.transform = transform
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # 1. 读取原始数据对 --------------------------------------------------------
        inp_tensor, gt_tensor = self.base[idx]  # 两个张量 ∈ [0,1]

        # 2. 物理预处理增强 --------------------------------------------------------
        # 转换张量到OpenCV格式 (HWC-BGR)
        inp_np = inp_tensor.permute(1, 2, 0).numpy()  # CHW → HWC
        inp_np = (inp_np * 255).astype(np.uint8)  # [0,1] → [0,255]
        inp_np = cv2.cvtColor(inp_np, cv2.COLOR_RGB2BGR)  # RGB → BGR

        # 执行传统增强流水线
        # pre_np = _physical_preprocess(inp_np)
        pre_np = inp_np

        # 3. 随机数据增强（珊瑚特征优化）---------------------------------------------
        # 随机锐化增强（30%概率）
        if random.random() < 0.1:
            pre_np = _random_sharpen(pre_np)

        # 随机色调偏移（50%概率，珊瑚颜色扰动）
        if random.random() < 0.1:
            pre_np = _random_hue_shift(pre_np)

        # 4. 格式转换与变换 --------------------------------------------------------
        # 转换回PIL格式
        pre_pil = Image.fromarray(cv2.cvtColor(pre_np, cv2.COLOR_BGR2RGB))
        gt_pil = self.to_pil(gt_tensor)

        # 应用统一变换
        if self.transform:
            pre = self.transform(pre_pil)
            gt = self.transform(gt_pil)
        else:
            pre = transforms.ToTensor()(pre_pil)
            gt = gt_tensor  # 保持原张量

        return pre, gt


# 传统增强函数（保持原有实现）
def enhance_color_correction(img: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(img)
    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
    mean_avg = (mean_r + mean_g + mean_b) / 3
    r = np.clip(r * (mean_avg / mean_r), 0, 255).astype(np.uint8)
    g = np.clip(g * (mean_avg / mean_g), 0, 255).astype(np.uint8)
    b = np.clip(b * (mean_avg / mean_b), 0, 255).astype(np.uint8)
    return cv2.merge((b, g, r))


def enhance_brightness(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


def enhance_sharpness(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)
