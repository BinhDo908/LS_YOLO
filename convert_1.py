import rasterio
import numpy as np
import os
import glob

DATASET_ROOT = r"D:\Training\Landslide_Dataset\Tài liệu ching chong"

print(f"{'Folder':<35} {'Sample file':<25} {'255%':>7} {'0%':>7} {'Cần INV?':>10}")
print("─" * 90)

for location in sorted(os.listdir(DATASET_ROOT)):
    label_dir = os.path.join(DATASET_ROOT, location, "label")
    if not os.path.isdir(label_dir):
        continue

    # Lấy 3 file random để check
    tifs = glob.glob(os.path.join(label_dir, "*.tif"))[:3]
    if not tifs:
        continue

    pct_255_list = []
    for tif in tifs:
        with rasterio.open(tif) as src:
            mask = src.read(1).astype(np.uint8)
        pct_255 = np.sum(mask == 255) / mask.size * 100
        pct_255_list.append(pct_255)

    avg_255 = np.mean(pct_255_list)
    avg_0   = 100 - avg_255
    need_inv = "✅ INV" if avg_255 > 50 else "❌ NORMAL"

    print(f"{location:<35} {os.path.basename(tifs[0]):<25} {avg_255:>6.1f}% {avg_0:>6.1f}% {need_inv:>10}")