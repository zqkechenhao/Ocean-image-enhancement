# compare_samples.py

import os
from PIL import Image
import matplotlib.pyplot as plt

# 这里填写你要对比的几个配置名
RUNS = [
    "tv0_vgg0.1",
    "tv1e-05_vgg0.1",
    "tv1e-04_vgg0.1",
    "tv1e-03_vgg0.1",
]
# 关注的 epoch 列表
EPOCHS = [5, 10, 15, 20]

fig, axes = plt.subplots(len(RUNS), len(EPOCHS), figsize=(4*len(EPOCHS), 4*len(RUNS)))
for i, run in enumerate(RUNS):
    for j, epoch in enumerate(EPOCHS):
        fname = f"samples/{run}_epoch{epoch}.png"
        if not os.path.isfile(fname):
            axes[i, j].axis('off')
            continue
        img = Image.open(fname)
        axes[i, j].imshow(img)
        axes[i, j].axis('off')
        if i == 0:
            axes[i, j].set_title(f"Epoch {epoch}", fontsize=16)
    axes[i, 0].set_ylabel(run, fontsize=16)

plt.tight_layout()
plt.show()
