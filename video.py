import cv2
import numpy as np
import rasterio
from pathlib import Path

IMG_DIR = r"D:\Training\Landslide_Dataset\Tài liệu ching chong\Hokkaido Iburi-Tobu\img"
OUT     = r"D:\hokkaido_birdseye.mp4"

imgs = sorted(Path(IMG_DIR).glob("*.tif"))[:100]  # lấy 100 ảnh đầu

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(OUT, fourcc, 5, (512, 512))

for p in imgs:
    with rasterio.open(str(p)) as src:
        img = np.stack([src.read(i) for i in range(1, 4)], axis=-1)
    img_u8 = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
    out.write(cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))

out.release()
print(f"Saved: {OUT}")