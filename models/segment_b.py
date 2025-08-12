# segmenter.py
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class SAMSegmenter:
    def __init__(self, checkpoint: str, model_type: str="vit_b"):
        # 加载 SAM 模型
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint).to("cuda")
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    @torch.no_grad()
    def get_mask(self, rgb: np.ndarray):
        """
        输入：H×W×3 uint8 RGB 图
        输出：H×W 二值掩码（前景物体）
        """
        masks = self.mask_generator.generate(rgb)
        # 多个 mask 取并集
        union = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        for m in masks:
            union |= m["segmentation"].astype(np.uint8)
        return union  # 0/1


if __name__ == "__main__":
    import cv2
    seg = SAMSegmenter(checkpoint="sam_vit_b_01ec64.pth")
    img = cv2.imread("test.jpg")[:,:,::-1]
    mask = seg.get_mask(img)
    cv2.imwrite("mask.png", mask*255)
