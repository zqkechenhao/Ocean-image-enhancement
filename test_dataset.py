# # test_dataset.py
#
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from RSGUnet_proj.dataset.euvp_dataset import EUVPPairedDataset
#
#
# def main():
#     # 定义 transform：先短边缩放到 286，再随机裁剪 256×256，最后 ToTensor
#     transform = transforms.Compose([
#         transforms.Resize(286),               # 短边缩放到 286，另一边等比放大
#         transforms.RandomCrop((256, 256)),    # 随机裁剪出 256×256
#         transforms.ToTensor(),                # HWC->[0,1] CHW
#     ])
#
#     ds = EUVPPairedDataset(
#         root_dir=r'D:\PycharmProjects\imageEnhance\RSGUnet_proj\dataset\EUVP',
#         split='train',
#         transform=transform
#     )
#     loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
#
#     for xin, xout in loader:
#         print("输入 batch shape:", xin.shape)   # 应该都是 [4,3,256,256]
#         print("目标 batch shape:", xout.shape)
#         break
#
#
# if __name__ == '__main__':
#     from torch.multiprocessing import freeze_support
#     freeze_support()
#     main()
# test_path.py
import os

mask_dir = "RSGUnet_proj/dataset/EUVP/Paired-20250414T045314Z-001/sam_masks_b/underwater_scenes/trainA"
mask_files = sorted(os.listdir(mask_dir))

print(f"Mask count: {len(mask_files)}")
for i, f in enumerate(mask_files[:20]):
    print(f"[{i}] {f}")


