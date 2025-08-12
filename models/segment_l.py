# models/segmenter.py

import os
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class SAMSegmenter:
    def __init__(self, checkpoint: str, model_type: str = "vit_l", device: str = "cuda"):
        # 1. 初始化设备
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # 2. 加载 SAM 模型
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(self.device)
        # 3. 创建自动分割器，参数可根据显存和目标大小调整
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=64,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_overlap_ratio=0.1
        )

    @torch.no_grad()
    def get_mask(self, rgb: np.ndarray) -> np.ndarray:
        """
        输入： H×W×3 uint8 RGB 图
        输出： H×W 二值掩模（前景物体）
        """
        # 直接调用生成器
        masks = self.mask_generator.generate(rgb)
        # 如果想把所有分割块合并为一个前景掩模，就把它们 OR 起来
        union = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        for m in masks:
            union |= (m["segmentation"].astype(np.uint8))
        return union  # 二值图，0/1


if __name__ == "__main__":
    # ------------------ 测试代码 ------------------ #
    # 路径按你的实际情况改
    SAM_CHECKPOINT = "sam_vit_l_0b3195.pth"
    IMG_PATH       = "test.jpg"
    OUT_MASK_PATH  = "mask.png"

    # 1. 实例化分割器
    seg = SAMSegmenter(checkpoint=SAM_CHECKPOINT, model_type="vit_l", device="cuda")

    # 2. 读一张测试图（BGR -> RGB）
    img_bgr = cv2.imread(IMG_PATH)
    img_rgb = img_bgr[:, :, ::-1]  # 转成 RGB 顺序，uint8

    # 3. 生成掩模
    mask = seg.get_mask(img_rgb)  # H×W, 值为 {0,1}

    # 4. 保存可视化：白底掩模
    cv2.imwrite(OUT_MASK_PATH, mask * 255)
    print(f"Saved binary mask to {OUT_MASK_PATH}")
