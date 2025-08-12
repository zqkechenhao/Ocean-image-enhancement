# callbacks.py
import os
import csv
import torch
from torchvision.utils import save_image
from utils import denorm


class TrainerCallbacks:
    def __init__(self, run_name, batch_size, output_dir="outputs"):
        self.run_name = run_name
        self.batch_size = batch_size
        self.output_dir = output_dir
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/samples/pre", exist_ok=True)
        os.makedirs(f"{output_dir}/samples/test/new", exist_ok=True)
        # 准备 CSV，包含额外的 laplacian 和 gray losses
        self.csv_path = f"{output_dir}/metrics_{run_name}.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.writer(f).writerow([
                'Epoch', 'PSNR', 'MSSSIM', 'GAN_G', 'GAN_D', 'UIQM', 'UCIQE', 'LAPLACIAN', 'GRAY'
            ])

    def save_checkpoint(self, state_dict, epoch):
        path = f"{self.output_dir}/checkpoints/rsgunet_new_color/{self.run_name}_epoch{epoch}.pth"
        torch.save(state_dict, path)

    def save_samples(self, inp, pred, tgt, epoch):
        # inp/pred/tgt 都为 tensor，[B,3,H,W]
        grid = torch.cat([denorm(inp), denorm(pred), denorm(tgt)], dim=0)
        path = f"{self.output_dir}/samples/test/{self.run_name}_epoch{epoch}.png"
        save_image(grid, path, nrow=self.batch_size, normalize=False)

    def log_metrics(self, epoch, psnr, ssim, ganG, ganD, uiqm, uciqe, lap, gray):
        # 写入 CSV
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{psnr:.3f}",
                f"{ssim:.4f}",
                f"{ganG:.4f}",
                f"{ganD:.4f}",
                f"{uiqm:.3f}",
                f"{uciqe:.3f}",
                f"{lap:.4f}",
                f"{gray:.4f}"
            ])
        # 打印日志
        print(
            f"[Epoch {epoch:03d}] PSNR={psnr:.3f}, SSIM={ssim:.4f}, "
            f"GAN_G={ganG:.4f}, GAN_D={ganD:.4f}, UIQM={uiqm:.3f}, UCIQE={uciqe:.3f}, "
            f"LAPLACIAN={lap:.4f}, GRAY={gray:.4f}"
        )
