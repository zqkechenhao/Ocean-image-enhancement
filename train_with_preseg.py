# train_with_preseg.py (增强版，含指标评估与CSV保存、模型保存)
import os, logging, csv, datetime
import torch
import torch.nn as nn
import kornia.color as kc
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.optim import Adam
from pytorch_msssim import ms_ssim

from models.unet_sam import UNetSAM
from models.discriminator import PatchDiscriminator
from RSGUnet_proj.VGG.vgg_feature_extractor import VGGPerceptualExtractor, compute_vgg_loss
from RSGUnet_proj.dataset.ocean_with_preseg import OceanPairedPreSeg
from utils import (
    get_loss_weights, edge_aware_loss, coral_color_loss, tv_loss,
    denorm, compute_psnr, uciqe, uiqm
)

logging.basicConfig(level=logging.INFO)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
LR = 5e-5
EPOCHS = 200

BASE = "RSGUnet_proj/dataset/EUVP/Paired-20250414T045314Z-001/"

RUN_NAME = "unet_preseg"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_DIR = f"samples_sam"
MODEL = f"checkpoints/unet_sam"
CSV_PATH = os.path.join(OUT_DIR, f"metrics_{RUN_NAME}.csv")
MODEL_SAVE_PATH = os.path.join(MODEL, f"best_model_{RUN_NAME}.pth")

os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # 数据增强变换
    tf_A = transforms.Compose([
        transforms.Resize(286),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*4, [0.5]*4)
    ])
    tf_B = transforms.Compose([
        transforms.Resize(286),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # 训练集和验证集
    train_ds = OceanPairedPreSeg(
        paired_root=os.path.join(BASE, "Paired"),
        mask_root=os.path.join(BASE, "sam_masks_b"),
        split="train",
        transform_A=tf_A, transform_B=tf_B)
    val_ds = OceanPairedPreSeg(
        paired_root=os.path.join(BASE, "Paired"),
        mask_root=os.path.join(BASE, "sam_masks_b"),
        split="validation",
        transform_A=tf_A, transform_B=tf_B)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = UNetSAM(in_ch=4).to(DEVICE)
    disc  = PatchDiscriminator(3).to(DEVICE)
    vgg   = VGGPerceptualExtractor(DEVICE).to(DEVICE)

    optG = Adam(model.parameters(), lr=LR)
    optD = Adam(disc.parameters(), lr=LR*0.5)

    best_psnr = 0.0

    # 初始化 CSV 文件
    with open(CSV_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'PSNR', 'SSIM', 'UIQM', 'UCIQE'])

    for epoch in range(EPOCHS):
        model.train()
        for inp4, tgt in train_loader:
            inp4, tgt = inp4.to(DEVICE), tgt.to(DEVICE)
            rgb = inp4[:, :3]
            pred = model(inp4)
            pred01 = (pred + 1) * 0.5
            tgt01 = (tgt + 1) * 0.5

            loss_l1 = nn.functional.l1_loss(pred01, tgt01)
            loss_vgg = compute_vgg_loss(vgg, pred01, tgt)
            loss_ssim = 1 - ms_ssim(pred01, tgt01, data_range=1.0)
            loss_edge = edge_aware_loss(pred, tgt)
            loss_tv = tv_loss(pred01)
            loss_col = coral_color_loss(pred, tgt)
            lab_pred = kc.rgb_to_lab(pred01)  # [B,3,H,W]
            lab_tgt = kc.rgb_to_lab(tgt01)
            mask = inp4[:, 3:4]
            ab_pred = lab_pred[:, 1:, :, :] * mask  # shape [B,2,H,W]
            ab_tgt = lab_tgt[:, 1:, :, :] * mask
            loss_ab = nn.functional.l1_loss(ab_pred, ab_tgt)
            loss_adv = -disc(rgb, pred01).mean()

            # laplacian loss
            lap_kernel = torch.tensor(
                [[[[0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]]]],
                dtype=pred01.dtype,
                device=pred01.device
            ).repeat(3, 1, 1, 1)
            lap_pred = nn.functional.conv2d(pred01, lap_kernel, padding=1, groups=3)
            lap_tgt = nn.functional.conv2d(tgt01, lap_kernel, padding=1, groups=3)
            loss_lap = nn.functional.l1_loss(lap_pred, lap_tgt)

            # sobel loss
            sobel_x = torch.tensor(
                [[[-1., 0., 1.],
                  [-2., 0., 2.],
                  [-1., 0., 1.]]],
                device=pred01.device, dtype=pred01.dtype
            )
            sobel_y = sobel_x.transpose(-1, -2)
            kx = sobel_x.repeat(3, 1, 1, 1)
            ky = sobel_y.repeat(3, 1, 1, 1)
            grad_pred = torch.sqrt(
                nn.functional.conv2d(pred01, kx, padding=1, groups=3) ** 2 +
                nn.functional.conv2d(pred01, ky, padding=1, groups=3) ** 2 + 1e-6
            )
            grad_tgt = torch.sqrt(
                nn.functional.conv2d(tgt01, kx, padding=1, groups=3) ** 2 +
                nn.functional.conv2d(tgt01, ky, padding=1, groups=3) ** 2 + 1e-6
            )
            loss_sobel = nn.functional.l1_loss(grad_pred, grad_tgt)

            # gray loss
            gray_weights = torch.tensor([0.299, 0.587, 0.114],
                                        device=pred01.device
                                        ).view(1, 3, 1, 1)
            gray_pred = (pred01 * gray_weights).sum(dim=1, keepdim=True)
            gray_tgt = (tgt01 * gray_weights).sum(dim=1, keepdim=True)
            loss_gray = nn.functional.l1_loss(gray_pred, gray_tgt)

            weights = get_loss_weights(epoch)
            lossG = (
                    weights['ssim'] * loss_ssim +
                    weights['vgg'] * loss_vgg +
                    weights['edge'] * loss_edge +
                    weights['edge_aware'] * loss_edge +
                    weights['tv'] * loss_tv +
                    weights['color'] * loss_col +
                    1.0 * loss_ab  +
                    weights['adv'] * loss_adv +
                    weights.get('lap', 0) * loss_lap +
                    weights.get('gray', 0) * loss_gray +
                    weights.get('sobel', 0) * loss_sobel +
                    0.5 * loss_l1
            )

            optG.zero_grad()
            lossG.backward()
            optG.step()

            # 判别器训练
            real_pred = disc(rgb, tgt01)
            fake_pred = disc(rgb, pred01.detach())
            loss_D = 0.5 * (torch.relu(1 - real_pred).mean() + torch.relu(1 + fake_pred).mean())
            optD.zero_grad()
            loss_D.backward()
            optD.step()

        # 保存训练样本图片
        # —— 保存训练样本对比 —— #
        model.eval()
        with torch.no_grad():
            # 从 val_loader 取一批
            inp4, orig3, tgt3 = next(iter(val_loader))
            inp4 = inp4.to(DEVICE)  # [B,4,H,W]
            orig3 = orig3.to(DEVICE)  # [B,3,H,W]
            tgt3 = tgt3.to(DEVICE)  # [B,3,H,W]

            # 模型输出完整增强图
            enhanced = model(inp4)  # [B,3,H,W], in [-1,1]
            enhanced01 = denorm(enhanced).clamp(0, 1)

            # 模型“看到”的 RGB（A4 前 3 通道）
            seen_rgb = denorm(inp4[:, :3]).clamp(0, 1)

            # 拼成 4 列：原图｜网络看到的｜增强后｜目标
            to_save = torch.cat([
                orig3[:4],
                seen_rgb[:4],
                enhanced01[:4],
                tgt3[:4]
            ], dim=0)

            # 每行 4 张
            save_image(to_save,
                       f"{OUT_DIR}/compare_epoch{epoch:03d}.png",
                       nrow=4, normalize=False)
        model.train()

        # 验证集评估
        model.eval()
        psnr_sum = ssim_sum = uiqm_sum = uciqe_sum = 0.0
        batches = 0

        with torch.no_grad():
            for inp4, tgt in val_loader:
                inp4, tgt = inp4.to(DEVICE), tgt.to(DEVICE)
                pred = model(inp4)
                pred01 = (pred + 1) * 0.5
                tgt01  = (tgt + 1) * 0.5

                psnr_sum += compute_psnr(pred01, tgt01)
                ssim_sum += ms_ssim(pred01, tgt01, data_range=1.0)
                uiqm_sum += uiqm(pred01)
                uciqe_sum += uciqe(pred01)
                batches += 1

        psnr_val  = psnr_sum / batches
        ssim_val  = ssim_sum / batches
        uiqm_val  = uiqm_sum / batches
        uciqe_val = uciqe_sum / batches

        # 记录指标到CSV
        with open(CSV_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{psnr_val:.3f}", f"{ssim_val:.4f}", f"{uiqm_val:.3f}", f"{uciqe_val:.3f}"])

        logging.info(f"[Epoch {epoch:03d}] PSNR={psnr_val:.3f}, SSIM={ssim_val:.4f}, UIQM={uiqm_val:.3f}, UCIQE={uciqe_val:.3f}")

        # 保存最优模型
        if psnr_val > best_psnr:
            best_psnr = psnr_val
            torch.save(model.state_dict(), MODEL)
            logging.info(f"Saved best model at epoch {epoch} with PSNR {psnr_val:.3f}")


if __name__ == "__main__":
    main()
