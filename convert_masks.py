import os
import cv2
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm

DATASET_ROOT = "D:\\Training\\Landslide_Dataset\\Tài liệu ching chong"
OUTPUT_LABELS = "D:\\Training\\labels_yolo"
MIN_AREA = 100

os.makedirs(OUTPUT_LABELS, exist_ok=True)

locations = [d for d in Path(DATASET_ROOT).iterdir() if d.is_dir()]
total, skipped = 0, 0

for loc in locations:
    label_dir = loc / "label"
    if not label_dir.exists():
        continue

    out_dir = Path(OUTPUT_LABELS) / loc.name
    out_dir.mkdir(exist_ok=True)

    masks = list(label_dir.glob("*.tif")) + list(label_dir.glob("*.TIF"))

    for mask_path in tqdm(masks, desc=loc.name):
        try:
            # Dùng rasterio thay vì cv2 để đọc được unicode path
            with rasterio.open(str(mask_path)) as src:
                mask = src.read(1).astype(np.uint8)
        except Exception as e:
            skipped += 1
            continue

        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        h, w = binary.shape

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lines = []
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        txt_path = out_dir / (mask_path.stem + ".txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))
        total += 1

print(f"\nDone! Converted: {total} | Skipped: {skipped}")