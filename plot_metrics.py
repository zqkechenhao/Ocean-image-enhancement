# plot_metrics.py
import os
import csv
import matplotlib.pyplot as plt


def load_metrics(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = {key: [] for key in headers}
        for row in reader:
            for key, val in zip(headers, row):
                try:
                    data[key].append(float(val))
                except ValueError:
                    data[key].append(val)
        return data


def plot_metrics(metrics, out_path="metrics_plot.png"):
    plt.figure(figsize=(18, 10))
    keys = list(metrics.keys())
    x = metrics['Epoch']

    subplot_keys = [
        ['PSNR', 'MSSSIM', 'UIQM', 'UCIQE'],
        ['GAN_G', 'GAN_D'],
        ['Loss_Lap', 'Loss_Gray'],
    ]

    for i, group in enumerate(subplot_keys, 1):
        plt.subplot(2, 2, i)
        for k in group:
            if k in metrics:
                plt.plot(x, metrics[k], label=k)
        plt.xlabel("Epoch")
        plt.title(f"Metrics: {', '.join(group)}")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[✓] 图像保存到 {out_path}")


if __name__ == '__main__':
    csv_file = os.path.join("samples_sam", "metrics_unet_preseg.csv")  # 修改为你实际的文件名
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"未找到 CSV 文件：{csv_file}")
    metrics = load_metrics(csv_file)
    plot_metrics(metrics)
