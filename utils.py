
import random
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import color
from torchvision import transforms


# --------------------------------- 数据增强 --------------------------------- #


# 珊瑚数据增强（集成到transform）
class CoralAugment:
    """珊瑚纹理与色调专属随机增强"""
    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        # 随机纹理卷积
        if random.random() < 0.3:
            kernels = [
                np.array([[0,1,0],[1,1,1],[0,1,0]]), np.array([[1,0,1],[0,1,0],[1,0,1]]),
                np.ones((3,3)), np.array([[2,1,0],[1,3,1],[0,1,2]]),
                np.array([[0,1,2],[1,3,1],[2,1,0]]), np.array([[0,2,0],[2,4,2],[0,2,0]]),
                np.array([[1,0,0],[0,3,0],[0,0,1]])
            ]
            k = random.choice(kernels).astype(np.float32)
            k /= k.sum()
            arr = cv2.filter2D(arr, -1, k)
        # 随机色调偏移
        if random.random() < 0.6:
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            hsv[:,:,0] = (hsv[:,:,0].astype(int) + random.choice([5,10,-5,-3])) % 180
            arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(arr)


# --------------------------------- 边缘增强 --------------------------------- #
# 在模型定义处确认边缘增强层的结构（模拟珊瑚年轮生长）
class EdgeEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # 第一层：捕捉海星触手的放射状纹路
            nn.Conv2d(64, 64, 3, padding=1),  # [0]
            nn.InstanceNorm2d(64),  # [1]
            nn.LeakyReLU(0.2),  # [2]
            # 第二层：强化水母触须的丝状边缘
            nn.Conv2d(64, 3, 1)  # [3]
        )

    def forward(self, x):
        return torch.clamp(x + self.conv(x), -1, 1)


# --------------------------------- 损失函数 --------------------------------- #


adv_criterion = nn.BCEWithLogitsLoss()


# Sobel 边缘一致性
def edge_aware_loss(pred, target):
    # build a 1×1 Sobel kernel
    sobel = torch.tensor([[[[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]]]],
                         device=pred.device)  # shape [1,1,3,3]
    # repeat it for each channel and do groups=3
    C = pred.shape[1]
    kernel = sobel.repeat(C, 1, 1, 1)           # [3,1,3,3]
    pred_grad = F.conv2d(pred, kernel, groups=C, padding=1)
    targ_grad = F.conv2d(target, kernel, groups=C, padding=1)
    # then compare
    return F.l1_loss(pred_grad, targ_grad)


# PSNR
def compute_psnr(pred, target):
    # pred, target in [-1,1] → map back to [0,1]
    pred01, target01 = (pred + 1) * 0.5, (target + 1) * 0.5
    mse = F.mse_loss(pred01, target01)
    return torch.tensor(100.0, device=pred.device) if mse == 0 else 10 * torch.log10(1.0 / mse)


def tv_loss(x):
    h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return h + w


# 珊瑚颜色校正损失（红通道权重1.5倍）
def coral_color_loss(pred, target):
    pred01 = (pred + 1) * 0.5
    target01 = (target + 1) * 0.5

    # 分别算三通道差异，加权后平均
    red_diff = F.l1_loss(pred01[:, 0:1], target01[:, 0:1]) * 1.2
    green_diff = F.l1_loss(pred01[:, 1:2], target01[:, 1:2]) * 0.9
    blue_diff = F.l1_loss(pred01[:, 2:3], target01[:, 2:3]) * 1.0

    return (1.1 * red_diff + green_diff + blue_diff) / 3


def laplacian_loss(pred, target):
    """高频（Laplacian）损失，增强细节保留"""
    # pred, target in [0,1]
    down_p = F.avg_pool2d(pred, 2)
    up_p   = F.interpolate(down_p, scale_factor=2, mode='bilinear', align_corners=False)
    hf_p   = pred - up_p
    down_t = F.avg_pool2d(target, 2)
    up_t   = F.interpolate(down_t, scale_factor=2, mode='bilinear', align_corners=False)
    hf_t   = target - up_t
    return F.l1_loss(hf_p, hf_t)


def filter_optim_state(optimizer, ckpt_state):
    """模拟珊瑚的共生过滤机制"""
    current_params = {id(p): p for p in optimizer.param_groups[0]['params']}
    filtered_state = {
        k: v for k, v in ckpt_state['state'].items()
        if id(v) in current_params
    }

    return {
        'state': filtered_state,
        'param_groups': ckpt_state['param_groups']
    }


# 白平衡
def simple_wb(x):
    # x: [B,3,H,W], 已经在[-1,1]
    x01, alpha = (x+1)*0.5, 0.5
    mean_rgb = x01.mean(dim=[2, 3], keepdim=True)  # [B,3,1,1]
    scale = 0.5*(1 + mean_rgb.mean(1,keepdim=True)/(mean_rgb+1e-6))
    scale = 1.0 + alpha * (scale - 1.0)
    return torch.clamp((x01 * scale)*2-1, -1, 1)





# UCIQE and UIQM from your definitions
def uciqe(img_tensor):
    batch, total = img_tensor.size(0), 0.0
    for i in range(batch):
        img = (img_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        _, S, V = cv2.split(hsv)
        cs = np.std(S) / 255.0
        sob = cv2.Sobel(V, cv2.CV_64F, 1, 1, ksize=5)
        cl = np.mean(np.abs(sob)) / 255.0
        sm = np.mean(S) / 255.0
        total += 0.4680 * cs + 0.2745 * cl + 0.2575 * sm
    return total / batch


def uiqm(img_tensor):
    batch, total = img_tensor.size(0), 0.0
    for i in range(batch):
        img = (img_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        lab = color.rgb2lab(img)
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        chroma = np.sqrt(A ** 2 + B ** 2)
        uicm = np.mean(chroma) - 0.3 * np.std(chroma)
        sobx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)
        soby = sobx.T
        gx = cv2.filter2D(L, -1, sobx, borderType=cv2.BORDER_REPLICATE)
        gy = cv2.filter2D(L, -1, soby, borderType=cv2.BORDER_REPLICATE)
        uism = np.mean(np.maximum(np.abs(gx), np.abs(gy)))
        uiconm = np.std(L) / (np.mean(L) + 1e-6)
        total += 0.0285 * uicm + 0.2853 * uism + 3.5753 * uiconm
    return total / batch


def gradient_penalty(D, real, fake, device, lambda_gp=10.0):
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolates = D(interpolates)
    grads = grad(
        outputs=d_interpolates.sum(), inputs=interpolates,
        create_graph=True, retain_graph=True
    )[0]
    grads = grads.view(B, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gp


def d_loss_wgan_gp(D, real, fake, device, lambda_gp=10.0):
    real_score = D(real)
    fake_score = D(fake)
    gp = gradient_penalty(D, real, fake, device, lambda_gp)
    # WGAN-GP discriminator loss: E[fake] - E[real] + λ GP
    return fake_score.mean() - real_score.mean() + gp, gp


def g_loss_wgan(D, fake):
    # WGAN generator loss: -E[D(fake)]
    return -D(fake).mean()


# 定义一个小的 Unsharp‑Mask
def unsharp_mask(x, amount=0.5):
    # x: [B,3,H,W] in [0,1]
    blurred = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    high_freq = x - blurred
    return torch.clamp(x + amount * high_freq, 0.0, 1.0)


# 权重函数
def get_loss_weights(epoch):
    """基于珊瑚礁纹理特征设计的渐进式权重策略"""
    adv_start, adv_ramp, adv_max = 60, 80, 0.0025
    adv = 0.0 if epoch < adv_start else min(adv_max, adv_max * (epoch - adv_start) / adv_ramp)
    return {

        'ssim': 0.35,
        'sobel': 0.01,
        'lap': 0.0 if epoch < 40 else 0.01,
        'gray': 0.0 if epoch < 40 else 0.02,
        'edge': 0.0 if epoch < 30 else 1e-5,
        'vgg': 0.3 + 0.1 * (epoch >= 20),  # 中期加强特征
        'edge_aware': 0.02 if epoch >= 30 else 0.0,
        'color': max(0.25, 0.35 - epoch * 0.001),  # 更平缓的颜色衰减
        'tv': 0.0 if epoch < 50 else 0.002,  # 后期加入TV去噪
        'adv': adv,
    }

# def denorm(x): return x.mul(0.5).add(0.5).clamp(0, 1)

# def get_loss_weights(epoch):
#     adv_start, adv_ramp, adv_max = 60, 80, 0.0025
#     adv = 0.0 if epoch < adv_start else min(adv_max, adv_max * (epoch - adv_start) / adv_ramp)
#     return {
#         'ssim': 0.35,
#         'sobel': 0.015,
#         'lap': 0.01 if epoch >= 20 else 0.0,
#         'gray': 0.015 if epoch >= 20 else 0.0,
#         'edge': 2e-5 if epoch >= 20 else 0.0,
#         'vgg': 0.4 if epoch >= 20 else 0.3,
#         'edge_aware': 0.03 if epoch >= 20 else 0.0,
#         'color': max(0.25, 0.35 - epoch * 0.001),
#         'tv': 0.002 if epoch >= 50 else 0.0,
#         'adv': adv,
#     }


def denorm(x): return (x + 1.0) * 0.5


