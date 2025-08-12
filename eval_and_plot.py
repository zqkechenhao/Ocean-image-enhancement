# eval_and_plot.py

import os, glob
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from pytorch_msssim import ms_ssim
from multiprocessing import freeze_support

from models.rsgunet import RSGUNet
from RSGUnet_proj.dataset.ocean_dataset import OceanPairedDataset
from utils import denorm, compute_psnr, uiqm, uciqe

from torch.utils.data import Dataset
from PIL import Image

# —— 配置区 —— #
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT   = 'checkpoints/rsgunet_new/rsgunet_new_epoch212.pth'  # <-- 换成你挑好的
OUT_DIR      = 'eval_results'
NUM_SAMPLES  = 30   # 前 N 张做可视化和曲线
BATCH_SIZE   = 1


class PairedFolderDataset(Dataset):
    """
    读入两个文件夹 A 和 B 中同名的图，返回 A->B 对
    """
    def __init__(self, dir_A, dir_B, transform=None):
        self.A_paths = sorted(glob.glob(os.path.join(dir_A, "*.*")))
        self.B_paths = sorted(glob.glob(os.path.join(dir_B, "*.*")))
        assert len(self.A_paths) == len(self.B_paths), "A/B 数量不匹配！"
        # 再校验同名
        for a, b in zip(self.A_paths, self.B_paths):
            assert os.path.basename(a) == os.path.basename(b), f"文件名不对应：{a} vs {b}"
        self.transform = transform

    def __len__(self):
        return len(self.A_paths)

    def __getitem__(self, idx):
        img_A = Image.open(self.A_paths[idx]).convert("RGB")
        img_B = Image.open(self.B_paths[idx]).convert("RGB")
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        return img_A, img_B


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/images", exist_ok=True)
    os.makedirs(f"{OUT_DIR}/plots", exist_ok=True)

    # —— 加载模型 —— #
    net = RSGUNet().to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()

    # —— 数据集&Loader —— #
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    # val_ds = OceanPairedDataset(
    #     root_dir='RSGUnet_proj/dataset/EUVP',
    #     split='val',
    #     transform=transform
    # )
    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=0,    # Windows 下设为 0，避免 spawn 错误
    #     pin_memory=True
    # )
    # —— 用 TrainA/TrainB 做对比 —— #
    dir_A = 'RSGUnet_proj/dataset/EUVP/Paired-20250414T045314Z-001/Paired/underwater_dark/trainA'
    dir_B = 'RSGUnet_proj/dataset/EUVP/Paired-20250414T045314Z-001/Paired/underwater_dark/trainB'
    paired_ds = PairedFolderDataset(dir_A, dir_B, transform=transform)
    val_loader = DataLoader(paired_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # —— 存放所有样本的指标 —— #
    psnr_A, psnr_P, psnr_B   = [], [], []
    ssim_A, ssim_P, ssim_B   = [], [], []
    uiqm_A, uiqm_P, uiqm_B   = [], [], []
    uciqe_A, uciqe_P, uciqe_B = [], [], []

    # —— 遍历前 N 张 —— #
    with torch.no_grad():
        for idx, (inp, tgt) in enumerate(val_loader):
            if idx >= NUM_SAMPLES:
                break

            inp = inp.to(DEVICE)
            tgt = tgt.to(DEVICE)

            pred = net(inp)

            # 归一化到 [0,1]
            A01 = denorm(inp)
            P01 = denorm(pred)
            B01 = denorm(tgt)

            # —— 保存三联图 —— #
            grid = torch.cat([A01, P01, B01], dim=0)
            save_image(
                grid,
                f"{OUT_DIR}/images/Train/sample_{idx:02d}.png",
                nrow=1,
                normalize=False
            )

            # —— 计算指标 —— #
            # PSNR
            psnr_A.append(compute_psnr(inp, tgt).item())
            psnr_P.append(compute_psnr(pred, tgt).item())
            psnr_B.append(100.0)  # B vs B → 理论上 ∞，此处用大值占位

            # SSIM (1 - ms_ssim)
            ssim_A.append((1 - ms_ssim(A01, B01, data_range=1.0)).item())
            ssim_P.append((1 - ms_ssim(P01, B01, data_range=1.0)).item())
            ssim_B.append(1.0)

            # UIQM / UCIQE
            uiqm_A.append(uiqm(inp).item())
            uiqm_P.append(uiqm(pred).item())
            uiqm_B.append(uiqm(tgt).item())

            uciqe_A.append(uciqe(inp).item())
            uciqe_P.append(uciqe(pred).item())
            uciqe_B.append(uciqe(tgt).item())

    print("Finished inference and metrics collection.")

    # —— 画曲线并保存 —— #
    def plot_metric(name, A_vals, P_vals, B_vals):
        plt.figure(figsize=(6, 4))
        x = list(range(len(A_vals)))
        plt.plot(x, A_vals, label='Input A → B', linestyle='--')
        plt.plot(x, P_vals, label='Predicted → B', linewidth=2)
        plt.plot(x, B_vals, label='GroundTruth B → B', linestyle=':')
        plt.title(name)
        plt.xlabel('Sample Index')
        plt.ylabel(name)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/plots/Train/{name.replace(' ', '_')}.png")
        plt.close()

    plot_metric('PSNR (dB)', psnr_A, psnr_P, psnr_B)
    plot_metric('SSIM',      ssim_A, ssim_P, ssim_B)
    plot_metric('UIQM',      uiqm_A, uiqm_P, uiqm_B)
    plot_metric('UCIQE',     uciqe_A, uciqe_P, uciqe_B)

    print(f"All plots saved under {OUT_DIR}/plots/Train")


if __name__ == '__main__':
    freeze_support()  # Windows 多进程安全启动
    main()
