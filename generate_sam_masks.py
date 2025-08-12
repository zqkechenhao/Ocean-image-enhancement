# generate_sam_masks.py
import os
import cv2
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import torch

# 配置
SAM_TYPE = "vit_h"
SAM_CKPT = "models/sam_vit_h_4b8939.pth"
INPUT_ROOT = "RSGUnet_proj/dataset/EUVP/Paired-20250414T045314Z-001/Paired"
OUTPUT_MASK_DIR = "RSGUnet_proj/dataset/EUVP/Paired-20250414T045314Z-001/Paired/sam_masks_h"

# 初始化 SAM
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT).to(device)
mask_gen = SamAutomaticMaskGenerator(sam)

# 创建输出目录
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# # 遍历 Paired 子目录 trainA
# for subdir in os.listdir(INPUT_ROOT):
#     trainA_dir = os.path.join(INPUT_ROOT, subdir, "trainA")
#     if not os.path.isdir(trainA_dir):
#         continue
#
#     output_subdir = os.path.join(OUTPUT_MASK_DIR, subdir)
#     os.makedirs(output_subdir, exist_ok=True)
#
#     for name in tqdm(os.listdir(trainA_dir), desc=f"Processing {subdir}"):
#         img_path = os.path.join(trainA_dir, name)
#         img = cv2.imread(img_path)
#         if img is None:
#             continue
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         masks = mask_gen.generate(img_rgb)
#         union = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
#         for m in masks:
#             union |= m["segmentation"].astype(np.uint8)
#
#         mask_out_path = os.path.join(output_subdir, name)
#         cv2.imwrite(mask_out_path, union * 255)

# 遍历 Paired 子目录
for subdir in os.listdir(INPUT_ROOT):
    sub_path = os.path.join(INPUT_ROOT, subdir)
    val_dir = os.path.join(sub_path, "validation")

    if not os.path.isdir(val_dir):
        continue

    print(f"[SAM] Generating masks for validation in {val_dir} ...")

    out_val_dir = os.path.join(OUTPUT_MASK_DIR, subdir, "validation")
    os.makedirs(out_val_dir, exist_ok=True)

    for fname in os.listdir(val_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(val_dir, fname)
        out_path = os.path.join(out_val_dir, os.path.splitext(fname)[0] + ".png")

        if os.path.exists(out_path):
            continue  # 跳过已存在

        try:
            img = cv2.imread(img_path)[:, :, ::-1]  # BGR → RGB
            masks = mask_gen.generate(img)

            union = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            for m in masks:
                union |= m["segmentation"].astype(np.uint8)

            cv2.imwrite(out_path, union * 255)
        except Exception as e:
            print(f"[WARN] Failed on {img_path}: {e}")
