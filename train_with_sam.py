# train_with_sam.py

import os
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from pytorch_msssim import ms_ssim

from models.unet_sam import UNetSAM
from models.discriminator import PatchDiscriminator
from RSGUnet_proj.VGG.vgg_feature_extractor import VGGPerceptualExtractor, compute_vgg_loss
from RSGUnet_proj.dataset.ocean_with_sam import OceanPairedWithSeg
from callbacks import TrainerCallbacks

from utils import (
    get_loss_weights,
    edge_aware_loss,
    coral_color_loss,
    tv_loss,
    denorm,
    compute_psnr,
    uciqe,
    uiqm,
    unsharp_mask,
)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIRED_DIR = os.path.join(BASE_DIR, "imageEnhance", "RSGUnet_proj", "dataset", "EUVP", "Paired-20250414T045314Z-001", "Paired")
TRAIN_DIR  = os.path.join(PAIRED_DIR, "train")
VAL_DIR    = os.path.join(PAIRED_DIR, "val")
logging.basicConfig(level=logging.INFO)
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LR         = 1e-5
EPOCHS     = 200
RUN_NAME   = "unet_sam"
SAM_CKPT   = "models/sam_vit_b_01ec64.pth"
OUT_DIR    = "checkpoints"
SAMPLE_DIR = "samples_sam"
os.makedirs(OUT_DIR,    exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

# train_with_sam.py


def main():
    callbacks = TrainerCallbacks(RUN_NAME, BATCH_SIZE, output_dir="outputs")
    # Data
    # 针对 A+mask 的 4通道变换
    tf_A = transforms.Compose([
        transforms.Resize(286),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 4, [0.5] * 4),
    ])

    # 针对 B 的 3通道变换
    tf_B = transforms.Compose([
        transforms.Resize(286),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    # train_with_sam.py 里这样改
    ds_train = OceanPairedWithSeg(
        root_dir=PAIRED_DIR,
        split="train",
        sam_ckpt=SAM_CKPT,
        transform_A=tf_A,
        transform_B=tf_B
    )
    train_subset = torch.utils.data.Subset(ds_train, list(range(32)))
    ds_val = OceanPairedWithSeg(
        root_dir=PAIRED_DIR,
        split="validation",  # 这里改成 validation
        sam_ckpt=SAM_CKPT,
        transform_A=tf_A,
        transform_B=tf_B
    )

    loader_t = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        prefetch_factor=4,
        persistent_workers=True,  # 让 worker 持久驻留
        pin_memory=True
    )

    loader_v = DataLoader(
        ds_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        prefetch_factor=4,
        persistent_workers=True,
        pin_memory=True
    )

    # Model / Opt
    net  = UNetSAM(in_ch=4).to(DEVICE)
    disc = PatchDiscriminator(3).to(DEVICE)
    vgg  = VGGPerceptualExtractor(device=DEVICE).to(DEVICE)

    optG = Adam(net.parameters(), lr=LR, betas=(0.9, 0.999))
    optD = Adam(disc.parameters(), lr=LR*0.5, betas=(0.5, 0.999))

    for epoch in range(EPOCHS):
        net.train()
        w = get_loss_weights(epoch)

        for inp4, tgt in loader_t:
            inp4, tgt = inp4.to(DEVICE), tgt.to(DEVICE)
            # split mask-away for generator input
            rgb_mask = inp4[:, :3]        # for disc/vgg/losses
            seg_mask = inp4[:, 3:].unsqueeze(1)  # [B,1,H,W]

            # forward G
            pred = net(inp4)               # [B,3,H,W]
            pred01 = (pred + 1) * 0.5
            tgt01  = (tgt  + 1) * 0.5

            # losses
            loss_l1   = nn.functional.l1_loss(pred01, tgt01)
            loss_vgg  = compute_vgg_loss(vgg, pred01, tgt)
            loss_ssim = 1 - ms_ssim(pred01, tgt01, data_range=1.0)
            loss_edge = edge_aware_loss(pred, tgt)
            loss_tv   = tv_loss(pred01)
            loss_col  = coral_color_loss(pred, tgt)

            # adversarial G
            fake_for_G = disc(rgb_mask, pred01)
            loss_G_adv = - fake_for_G.mean() if w['adv']>0 else torch.tensor(0., device=DEVICE)

            lossG = (
                0.6 * loss_l1
                + w['vgg']   * loss_vgg
                + w['ssim']  * loss_ssim
                + w['edge']  * loss_edge
                + w['tv']    * loss_tv
                + w['color'] * loss_col
                + w['adv']   * loss_G_adv
            )

            optG.zero_grad()
            lossG.backward()
            optG.step()

            # update D
            if w['adv'] > 0:
                real_pred = disc(rgb_mask, tgt01)
                fake_pred = disc(rgb_mask, pred01.detach())
                loss_D = 0.5 * (torch.relu(1 - real_pred).mean() + torch.relu(1 + fake_pred).mean())
                optD.zero_grad()
                loss_D.backward()
                optD.step()

        # log & save one sample
        net.eval()
        with torch.no_grad():
            inp4, tgt = next(iter(loader_v))
            inp4, tgt = inp4.to(DEVICE), tgt.to(DEVICE)
            pred = net(inp4)
            grid = torch.cat([
                denorm(inp4[:, :3]),
                denorm((pred+1)*0.5),
                denorm((tgt+1)*0.5),
            ], dim=0)
            save_image(grid,
                       f"{SAMPLE_DIR}/{RUN_NAME}_ep{epoch:03d}.png",
                       nrow=BATCH_SIZE, normalize=False)

        # save checkpoint
        torch.save({
            'epoch': epoch,
            'net':   net.state_dict(),
            'optG':  optG.state_dict(),
            'optD':  optD.state_dict(),
        }, f"{OUT_DIR}/{RUN_NAME}_epoch{epoch:03d}.pth")

        # 评估验证集指标
        net.eval()
        psnr_sum = ssim_sum = ganG_sum = ganD_sum = uiqm_sum = uciqe_sum = lap_sum = gray_sum = 0.0
        batches = 0
        lap_kernel = torch.tensor(
            [[[[0, 1, 0],
               [1, -4, 1],
               [0, 1, 0]]]],
            dtype=torch.float32,
            device=DEVICE
        ).repeat(3, 1, 1, 1)
        gray_weights = torch.tensor([0.299, 0.587, 0.114], device=DEVICE).view(1, 3, 1, 1)

        with torch.no_grad():
            for inp4, tgt in loader_v:
                inp4, tgt = inp4.to(DEVICE), tgt.to(DEVICE)
                pred = net(inp4)
                pred01 = (pred + 1) * 0.5
                tgt01 = (tgt + 1) * 0.5

                psnr_sum += compute_psnr(pred, tgt)
                ssim_sum += 1 - ms_ssim(pred01, tgt01, data_range=1.0)
                ganG_sum += nn.functional.binary_cross_entropy_with_logits(disc(inp4[:, :3], pred01),
                                                                           torch.ones_like(tgt01[:, :1]))
                ganD_sum += nn.functional.binary_cross_entropy_with_logits(disc(inp4[:, :3], tgt01),
                                                                           torch.ones_like(tgt01[:, :1]))
                uiqm_sum += uiqm(pred)
                uciqe_sum += uciqe(pred)

                # laplacian
                lap_pred = nn.functional.conv2d(pred01, lap_kernel, padding=1, groups=3)
                lap_tgt = nn.functional.conv2d(tgt01, lap_kernel, padding=1, groups=3)
                lap_sum += nn.functional.l1_loss(lap_pred, lap_tgt).item()

                # gray
                gray_pred = (pred01 * gray_weights).sum(dim=1, keepdim=True)
                gray_tgt = (tgt01 * gray_weights).sum(dim=1, keepdim=True)
                gray_sum += nn.functional.l1_loss(gray_pred, gray_tgt).item()

                batches += 1

        # 写入 CSV
        callbacks.log_metrics(
            epoch,
            psnr_sum / batches,
            1 - (ssim_sum / batches),
            ganG_sum / batches,
            ganD_sum / batches,
            uiqm_sum / batches,
            uciqe_sum / batches,
            lap_sum / batches,
            gray_sum / batches
        )

        logging.info(f"Epoch {epoch:03d} | L1:{loss_l1:.4f} SSIM:{1-loss_ssim:.4f} | saved.")


if __name__ == "__main__":
    main()
