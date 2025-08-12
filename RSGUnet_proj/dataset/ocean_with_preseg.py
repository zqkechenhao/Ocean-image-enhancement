import os
from PIL import Image
from torch.utils.data import Dataset
import warnings


class OceanPairedPreSeg(Dataset):
    def __init__(self, paired_root, mask_root, split, transform_A, transform_B):
        super().__init__()
        self.paired_root = paired_root
        self.mask_root   = mask_root
        self.split       = split.lower()
        self.transform_A = transform_A
        self.transform_B = transform_B

        self.A_paths = []
        self.B_paths = []
        self.M_paths = []

        for scene in sorted(os.listdir(paired_root)):
            scene_dir = os.path.join(paired_root, scene)
            if not os.path.isdir(scene_dir):
                continue

            if self.split in ("val", "validation"):
                subA = "validation"
                subB = "validation"
                subM = "validation"
            else:
                subA = "trainA"
                subB = "trainB"
                subM = "trainA"

            dirA = os.path.join(scene_dir, subA)
            dirB = os.path.join(scene_dir, subB)
            mask_scene_dir = os.path.join(mask_root, scene, subM)

            if not (os.path.isdir(dirA) and os.path.isdir(dirB) and os.path.isdir(mask_scene_dir)):
                continue

            # 读取掩码文件名（不含扩展名）字典，加快模糊查找
            mask_files = {
                os.path.splitext(f)[0]: os.path.join(mask_scene_dir, f)
                for f in os.listdir(mask_scene_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            }

            for fn in sorted(os.listdir(dirA)):
                if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                name_no_ext = os.path.splitext(fn)[0]
                pA = os.path.join(dirA, fn)
                pB = os.path.join(dirB, fn)

                # 1) 精确匹配掩码文件名
                if name_no_ext in mask_files:
                    pM = mask_files[name_no_ext]
                else:
                    # 2) 尝试模糊匹配（前缀匹配）
                    candidates = [mask_path for k, mask_path in mask_files.items() if k.startswith(name_no_ext)]
                    if candidates:
                        pM = candidates[0]
                    else:
                        warnings.warn(f"[WARN] Skipped {fn} -> Mask not found for: {name_no_ext}")
                        continue

                if os.path.isfile(pA) and os.path.isfile(pB) and os.path.isfile(pM):
                    self.A_paths.append(pA)
                    self.B_paths.append(pB)
                    self.M_paths.append(pM)

        assert len(self.A_paths) > 0, f"No samples for split={split} in {paired_root}"

    def __len__(self):
        return len(self.A_paths)

    def __getitem__(self, idx):
        imgA = Image.open(self.A_paths[idx]).convert("RGB")
        imgB = Image.open(self.B_paths[idx]).convert("RGB")
        mask = Image.open(self.M_paths[idx]).convert("L")

        A4 = Image.merge("RGBA", (*imgA.split(), mask))
        A4 = self.transform_A(A4)
        B3 = self.transform_B(imgB)[:3]

        return A4, B3
