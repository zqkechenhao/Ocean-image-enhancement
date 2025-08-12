# dataset_analysis.py

import os
from PIL import Image
import numpy as np
import pandas as pd


def collect_image_stats(root_dirs):
    records = []
    for root in root_dirs:
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname.lower().endswith(('.jpg','.jpeg','.png')):
                    path = os.path.join(dirpath, fname)
                    try:
                        with Image.open(path) as img:
                            w, h = img.size
                    except Exception as e:
                        print(f"无法打开 {path}：{e}")
                        continue
                    records.append({
                        'path': path,
                        'width': w,
                        'height': h,
                        'ratio': w / h
                    })
    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    # 请根据你的目录结构修改下面的路径列表
    paired_dirs = [
        r"D:\PycharmProjects\imageEnhance\RSGUnet_proj\dataset\EUVP\Paired-20250414T045314Z-001\Paired\underwater_dark\trainA",
        r"D:\PycharmProjects\imageEnhance\RSGUnet_proj\dataset\EUVP\Paired-20250414T045314Z-001\Paired\underwater_dark\trainB",
        r"D:\PycharmProjects\imageEnhance\RSGUnet_proj\dataset\EUVP\Paired-20250414T045314Z-001\Paired\underwater_dark\validation",
        r"D:\PycharmProjects\imageEnhance\RSGUnet_proj\dataset\EUVP\Paired-20250414T045314Z-001\Paired\underwater_imagenet\trainA",
        # … 以后再加上其他子集路径
    ]

    df = collect_image_stats(paired_dirs)
    # 汇总统计
    summary = {
        'width_min': df['width'].min(),
        'width_max': df['width'].max(),
        'width_mean': df['width'].mean(),
        'height_min': df['height'].min(),
        'height_max': df['height'].max(),
        'height_mean': df['height'].mean(),
        'ratio_min': df['ratio'].min(),
        'ratio_max': df['ratio'].max(),
        'ratio_mean': df['ratio'].mean(),
        'num_images': len(df)
    }
    print("=== 分辨率统计 ===")
    for k, v in summary.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
    # 可视化长宽比分布（可选）
    try:
        import matplotlib.pyplot as plt
        plt.hist(df['ratio'], bins=50)
        plt.title("Aspect Ratio Distribution")
        plt.xlabel("width/height")
        plt.ylabel("count")
        plt.show()
    except ImportError:
        pass
