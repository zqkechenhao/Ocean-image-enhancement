import os
import csv

from datetime import datetime
from multiprocessing import freeze_support

import torch

import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from pytorch_msssim import ms_ssim

from models.NetWithColor import NetWithColor
from models.discriminator import PatchDiscriminator
from RSGUnet_proj.VGG.vgg_feature_extractor import VGGPerceptualExtractor, compute_vgg_loss
from RSGUnet_proj.dataset.ocean_dataset import OceanPairedDataset
from RSGUnet_proj.dataset.euvp_dataset import EUVPPairedDataset
from callbacks import TrainerCallbacks

from torch import ones_like, zeros_like
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy_with_logits

from utils import (
    CoralAugment,
    EdgeEnhancer,
    laplacian_loss,
    simple_wb,
    get_loss_weights,
    edge_aware_loss,
    coral_color_loss,
    tv_loss,
    denorm,
    compute_psnr,
    uciqe,
    uiqm,
    unsharp_mask
)

import logging

# --------------------------------- configuration --------------------------------- #
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
LR = 1e-5
EPOCHS = 301
RUN_NAME = f"rsgunet_new_color"
RESUME_CHECKPOINT = f"checkpoints/rsgunet_new_color/{RUN_NAME}_epoch0.pth"


# --------------------------------- main function --------------------------------- #
def main():
    callbacks = TrainerCallbacks(RUN_NAME, BATCH_SIZE, output_dir="outputs")
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples/pre', exist_ok=True)
    os.makedirs('samples/test', exist_ok=True)

    # 结果记录 CSV
    csv_path = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'Epoch', 'PSNR', 'MSSSIM', 'GAN_G', 'GAN_D', 'UIQM', 'UCIQE',
            'Loss_Lap', 'Loss_Gray'
        ])

    # 数据加载与变换
    transform = transforms.Compose([
        transforms.Resize(286),
        CoralAugment(),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    root = os.path.join(os.getcwd(), 'RSGUnet_proj', 'dataset', 'EUVP')
    train_ds = OceanPairedDataset(root, split='train', transform=transform)
    val_ds = OceanPairedDataset(root, split='val', transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 构建模型与优化器
    net = NetWithColor().to(DEVICE)
    disc = PatchDiscriminator(3).to(DEVICE)
    vgg_ext = VGGPerceptualExtractor(device=DEVICE).to(DEVICE)

    opt_G = Adam(net.parameters(), lr=LR, betas=(0.9, 0.999))
    opt_D = Adam(disc.parameters(), lr=LR * 0.5, betas=(0.5, 0.999))
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_G, T_0=40, T_mult=1, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    # 恢复训练
    start_epoch = 0
    if os.path.exists(RESUME_CHECKPOINT):
        ckpt = torch.load(RESUME_CHECKPOINT)
        net.load_state_dict(ckpt['net'], strict=False)
        opt_G.load_state_dict(ckpt['opt_G'])
        opt_D.load_state_dict(ckpt['opt_D'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        log.info(f"Resumed from epoch {ckpt['epoch']}.")
    else:
        log.warning(f"No checkpoint found at {RESUME_CHECKPOINT}, training from scratch.")

    # 训练循环
    for epoch in range(start_epoch, EPOCHS):
        net.train()
        w = get_loss_weights(epoch)

        for i, (inp, tgt) in enumerate(train_loader, 1):
            inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)

            # 前向推理
            with torch.cuda.amp.autocast():
                pred, ab_pred = net(inp)
                pred01 = (pred + 1) * 0.5
                tgt01 = (tgt + 1) * 0.5

                # 损失计算
                loss_l1 = F.l1_loss(pred01, tgt01)
                loss_vgg = compute_vgg_loss(vgg_ext, pred01, tgt)
                loss_ssim = 1 - ms_ssim(pred01, tgt01, data_range=1.0)
                loss_edge = edge_aware_loss(pred, tgt)
                loss_tv = tv_loss(pred01)
                loss_color = coral_color_loss(pred, tgt)
                loss_color_branch = net.color_loss(ab_pred, pred01)

                # laplacian loss
                lap_kernel = torch.tensor(
                    [[[[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]]]],
                    dtype=pred01.dtype,
                    device=pred01.device
                ).repeat(3, 1, 1, 1)

                lap_pred = F.conv2d(pred01, lap_kernel, padding=1, groups=3)
                lap_tgt = F.conv2d(tgt01, lap_kernel, padding=1, groups=3)
                loss_lap = F.l1_loss(lap_pred, lap_tgt)

                # ——— 新增 Sobel 边缘一致性损失 ———
                # 定义 Sobel 卷积核
                sobel_x = torch.tensor(
                    [[[-1., 0., 1.],
                      [-2., 0., 2.],
                      [-1., 0., 1.]]],
                    device=pred01.device, dtype=pred01.dtype
                )
                sobel_y = sobel_x.transpose(-1, -2)
                # 扩展到 3 通道
                kx = sobel_x.repeat(3, 1, 1, 1)
                ky = sobel_y.repeat(3, 1, 1, 1)
                # 计算梯度幅值
                grad_pred = torch.sqrt(
                    F.conv2d(pred01, kx, padding=1, groups=3) ** 2 +
                    F.conv2d(pred01, ky, padding=1, groups=3) ** 2 + 1e-6
                )
                grad_tgt = torch.sqrt(
                    F.conv2d(tgt01, kx, padding=1, groups=3) ** 2 +
                    F.conv2d(tgt01, ky, padding=1, groups=3) ** 2 + 1e-6
                )
                loss_sobel = F.l1_loss(grad_pred, grad_tgt)

                # ——— 新增：grayscale consistency loss ———
                gray_weights = torch.tensor([0.299, 0.587, 0.114],
                                            device=pred01.device
                                            ).view(1, 3, 1, 1)
                gray_pred = (pred01 * gray_weights).sum(dim=1, keepdim=True)
                gray_tgt = (tgt01 * gray_weights).sum(dim=1, keepdim=True)
                loss_gray = F.l1_loss(gray_pred, gray_tgt)

                # fake_pred = disc(inp, pred01)
                # valid = torch.ones_like(fake_pred)
                # loss_gan_G = F.binary_cross_entropy_with_logits(fake_pred, valid)

                # Generator 对抗项 (Hinge)
                if w['adv'] > 0:
                    # 重新过一遍 G 预测，不 detach
                    fake_for_G = disc(inp, pred01)
                    # Hinge 下的 G 对抗 loss
                    loss_G_adv = - fake_for_G.mean()
                else:
                    loss_G_adv = torch.tensor(0., device=DEVICE)


                loss_G = (
                        0.6 * loss_l1
                        + w['vgg'] * loss_vgg
                        + w['ssim'] * loss_ssim
                        + w['edge'] * loss_edge
                        + w['edge_aware'] * edge_aware_loss(pred, tgt)
                        + w['adv'] * loss_G_adv
                        + w['color'] * loss_color
                        + w['tv'] * loss_tv
                        + w.get('lap', 0) * loss_lap
                        + w.get('gray', 0) * loss_gray
                        + w.get('sobel', 0) * loss_sobel
                        + 0.05 * loss_color_branch
                )

            opt_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()
            scheduler_G.step(epoch + i / len(train_loader))

            # Discriminator
            if w['adv'] > 0:
                tgt01.requires_grad_(True)
                real_pred = disc(inp, tgt01)

                # real_labels = torch.full_like(real_pred, 0.9)
                # loss_real = F.binary_cross_entropy_with_logits(real_pred, real_labels)

                fake_pred = disc(inp, pred01.detach())
                # 假样本标签在 0.1
                # fake_labels = torch.full_like(fake_pred, 0.1)
                # loss_fake = F.binary_cross_entropy_with_logits(fake_pred, fake_labels)

                # hinge
                loss_D = 0.5 * (F.relu(1 - real_pred).mean() + F.relu(1 + fake_pred).mean())

                # R1 regularization
                grad_real = torch.autograd.grad(
                    outputs=real_pred.sum(),  # scalar
                    inputs=tgt01,  # require_grad=True
                    create_graph=True
                )[0]
                loss_r1 = (grad_real.view(grad_real.size(0), -1)
                           .norm(2, dim=1) ** 2
                           ).mean() * 10.0
                loss_D = loss_D + loss_r1

                opt_D.zero_grad()
                loss_D.backward()
                # 裁剪梯度，避免爆炸导致 NaN
                torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
                opt_D.step()

                # 训练完后，把 real 的 grad 钩子关掉
                tgt01.requires_grad_(False)

        # 保存样本与模型
        net.eval()
        sample_in, sample_tgt = next(iter(val_loader))
        with torch.no_grad():
            sample_pred, _ = net(sample_in.to(DEVICE))

        # 在生成 sample_pred 之后：
        sample_pred01 = (sample_pred + 1) * 0.5  # [-1,1] → [0,1]
        sample_sharp = unsharp_mask(sample_pred01, amount=0.7)

        sample_sharp = sample_sharp.cpu()

        save_image(torch.cat([denorm(sample_in), sample_sharp, denorm(sample_tgt)], dim=0),
                   f'samples/test/new/{RUN_NAME}_epoch{epoch}.png', nrow=BATCH_SIZE, normalize=False)

        torch.save({
            'epoch': epoch,
            'net': net.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_D': opt_D.state_dict(),
            'scaler': scaler.state_dict()
        }, f'checkpoints/rsgunet_new/{RUN_NAME}_epoch{epoch}.pth')

        # 记录指标
        # ——— 记录验证指标 & 写入 CSV ———
        net.eval()
        psnr_sum = ssim_sum = ganG_sum = ganD_sum = uiqm_sum = uciqe_sum = lap_sum = gray_sum = 0.0
        batches = 0

        with torch.no_grad():
            for inp_v, tgt_v in val_loader:
                inp_v, tgt_v = inp_v.to(DEVICE), tgt_v.to(DEVICE)
                pred_v, _ = net(inp_v)

                # 映射到 [0,1]
                pred01 = (pred_v + 1) * 0.5
                tgt01 = (tgt_v + 1) * 0.5

                # PSNR / SSIM
                psnr_sum += compute_psnr(pred_v, tgt_v)
                ssim_sum += 1 - ms_ssim(pred01, tgt01, data_range=1.0)

                # GAN 指标
                fake_pred_v = disc(inp_v, pred01)
                real_pred_v = disc(inp_v, tgt01)
                ganG_sum += binary_cross_entropy_with_logits(fake_pred_v, ones_like(fake_pred_v))
                ganD_sum += binary_cross_entropy_with_logits(real_pred_v, ones_like(real_pred_v))

                # UIQM / UCIQE
                uiqm_sum += uiqm(pred_v)
                uciqe_sum += uciqe(pred_v)

                # lap loss
                lap_pred = F.conv2d(pred01, lap_kernel, padding=1, groups=3)
                lap_tgt = F.conv2d(tgt01, lap_kernel, padding=1, groups=3)
                lap_sum += F.l1_loss(lap_pred, lap_tgt).item()
                # gray loss
                gray_pred = (pred01 * gray_weights).sum(dim=1, keepdim=True)
                gray_tgt = (tgt01 * gray_weights).sum(dim=1, keepdim=True)
                gray_sum += F.l1_loss(gray_pred, gray_tgt).item()

                batches += 1

        # 计算平均值
        psnr_val = psnr_sum / batches
        ssim_val = ssim_sum / batches
        ganG_val = ganG_sum / batches
        ganD_val = ganD_sum / batches
        uiqm_val = uiqm_sum / batches
        uciqe_val = uciqe_sum / batches
        lap_val = lap_sum / batches
        gray_val = gray_sum / batches

        # 写 CSV 并打印
        callbacks.log_metrics(
            epoch,
            psnr_val.item() if isinstance(psnr_val, torch.Tensor) else psnr_val,
            ssim_val.item() if isinstance(ssim_val, torch.Tensor) else ssim_val,
            ganG_val.item() if isinstance(ganG_val, torch.Tensor) else ganG_val,
            ganD_val.item() if isinstance(ganD_val, torch.Tensor) else ganD_val,
            uiqm_val,
            uciqe_val,
            lap_val,
            gray_val
        )

        log.info(f"Epoch {epoch:03d} complete. "
                 f"PSNR={psnr_val:.3f}, SSIM={1 - ssim_val:.4f}, "
                 f"GAN_G={ganG_val:.4f}, GAN_D={ganD_val:.4f}, "
                 f"UIQM={uiqm_val:.3f}, UCIQE={uciqe_val:.3f}")

        log.info(f"Epoch {epoch} complete.")
        avg_rgb = pred01.mean(dim=[0, 2, 3]).cpu().numpy()
        log.info(f"Avg RGB = R:{avg_rgb[0]:.3f}, G:{avg_rgb[1]:.3f}, B:{avg_rgb[2]:.3f}")

    log.info("Training finished.")


if __name__ == '__main__':
    freeze_support()
    main()
