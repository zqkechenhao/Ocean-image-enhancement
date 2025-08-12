import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


class OceanPairedWithSeg(Dataset):
    def __init__(self, root_dir, split, sam_ckpt,  transform_A=None, transform_B=None):
        super().__init__()
        self.transform_A = transform_A
        self.transform_B = transform_B
        self.pairs = []
        self.split = split.lower()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 遍历所有子目录
        for subfolder in os.listdir(root_dir):
            subdir = os.path.join(root_dir, subfolder)
            if not os.path.isdir(subdir):
                continue

            if self.split in ['val', 'validation']:
                dirA = os.path.join(subdir, 'validation')
                dirB = os.path.join(subdir, 'validation')
            else:
                dirA = os.path.join(subdir, 'trainA')
                dirB = os.path.join(subdir, 'trainB')

            if not (os.path.isdir(dirA) and os.path.isdir(dirB)):
                continue


            names = sorted(os.listdir(dirA))
            for name in names:
                pathA = os.path.join(dirA, name)
                pathB = os.path.join(dirB, name)
                if os.path.isfile(pathA) and os.path.isfile(pathB):
                    self.pairs.append((pathA, pathB))


        if len(self.pairs) == 0:
            raise RuntimeError(f"未找到任何 {split} 数据对，请检查目录结构！")

        print(f"[OceanPairedWithSeg] Loaded {len(self.pairs)} samples for split={split}")
        self._init_sam(sam_ckpt)

    def _init_sam(self, checkpoint):
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint).to(self.device)
        self.mask_gen = SamAutomaticMaskGenerator(self.sam)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        print(f"[Dataset] Loading sample {idx}")
        from PIL import Image
        import numpy as np

        pathA, pathB = self.pairs[idx]
        imgA = Image.open(pathA).convert("RGB")
        imgB = Image.open(pathB).convert("RGB")

        npA = np.array(imgA)
        masks = self.mask_gen.generate(npA)
        union = np.zeros((npA.shape[0], npA.shape[1]), dtype=np.uint8)
        for m in masks:
            union |= m["segmentation"].astype(np.uint8)
        mask = Image.fromarray(union * 255)

        if self.transform_A and self.transform_B:
            A4 = self.transform_A(Image.merge("RGBA", imgA.split() + (mask,)))
            B3 = self.transform_B(imgB.convert("RGB"))
        else:
            raise RuntimeError("Need transform_A and transform_B")

        return A4, B3

