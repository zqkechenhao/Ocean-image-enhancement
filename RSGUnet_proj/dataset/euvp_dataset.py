import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class EUVPPairedDataset(Dataset):
    """
    PyTorch Dataset for EUVP Paired subsets.
    Assumes directory structure:
      root_dir/
        Paired-*/
          Paired/
            <subset>/
              trainA/, trainB/, validation/
    """
    def __init__(self, root_dir, split='train', subsets=None, transform=None):
        super().__init__()
        # 找到 Paired-xxxx 目录
        paired_base = None
        for d in os.listdir(root_dir):
            if d.lower().startswith('paired-') and os.path.isdir(os.path.join(root_dir, d, 'Paired')):
                paired_base = os.path.join(root_dir, d, 'Paired')
                break
        if paired_base is None:
            raise RuntimeError(f"EUVP Paired directory not found in {root_dir}")

        # 确定拆分模式
        self.split = split.lower()
        if self.split not in ('train', 'val'):
            raise ValueError("split must be 'train' or 'val'")

        # 构建所有图像对列表
        self.inputs = []
        self.targets = []
        for subset in sorted(os.listdir(paired_base)):
            subdir = os.path.join(paired_base, subset)
            if not os.path.isdir(subdir):
                continue
            if self.split == 'train':
                dirA = os.path.join(subdir, 'trainA')
                dirB = os.path.join(subdir, 'trainB')
            else:
                dirA = os.path.join(subdir, 'validation')
                dirB = os.path.join(subdir, 'validation')
            if not os.path.isdir(dirA) or not os.path.isdir(dirB):
                continue
            for fname in sorted(os.listdir(dirA)):
                pathA = os.path.join(dirA, fname)
                pathB = os.path.join(dirB, fname)
                if os.path.exists(pathA) and os.path.exists(pathB):
                    self.inputs.append(pathA)
                    self.targets.append(pathB)

        if len(self.inputs) == 0:
            raise RuntimeError(f"No image pairs found for split {split} in {paired_base}")

        # 转换操作
        if transform is None:
            # 默认：ToTensor 将 [0,255] HWC -> [0,1] CHW
            transform = transforms.ToTensor()
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        img_in = Image.open(self.inputs[idx]).convert('RGB')
        img_out = Image.open(self.targets[idx]).convert('RGB')
        if self.transform:
            img_in = self.transform(img_in)
            img_out = self.transform(img_out)
        return img_in, img_out


# Usage example:
# from dataset.euvp_dataset import EUVPPairedDataset
# train_ds = EUVPPairedDataset(root_dir='dataset/EUVP', split='train')
# val_ds   = EUVPPairedDataset(root_dir='dataset/EUVP', split='val')
# loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
