import rasterio, numpy as np
from pathlib import Path

label_dir = Path('D:/Training/labels_yolo')
img_dir = Path('D:/Training/Landslide_Dataset/Tài liệu ching chong')

print("=== Mask coverage for files with bbox 1.0x1.0 ===")
count = 0
for txt in label_dir.rglob('*.txt'):
    lines = txt.read_text().strip().splitlines()
    if not lines:
        continue
    parts = lines[0].split()
    if float(parts[3]) >= 0.99 and float(parts[4]) >= 0.99:
        loc = txt.parent.name
        mask_path = img_dir / loc / 'label' / (txt.stem + '.tif')
        if mask_path.exists():
            with rasterio.open(str(mask_path)) as src:
                m = src.read(1)
            pct = (m > 0).mean() * 100
            print(f'{txt.name}: coverage={pct:.1f}%, unique_vals={np.unique(m).tolist()}, shape={m.shape}')
            count += 1
        if count >= 10:
            break
