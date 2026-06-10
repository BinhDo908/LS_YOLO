import numpy as np
import rasterio
import cv2
import os

IMG_PATH   = r"D:\Training\Landslide_Dataset\Tài liệu ching chong\Hokkaido Iburi-Tobu\img\Hokkaido0013.tif"
LABEL_PATH = r"D:\Training\Landslide_Dataset\Tài liệu ching chong\Hokkaido Iburi-Tobu\label\Hokkaido0013.tif"
OUT        = r"D:\conversion_result.png"

# Đọc ảnh gốc
with rasterio.open(IMG_PATH) as src:
    img = np.stack([src.read(i) for i in range(1, src.count+1)], axis=-1)
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Đọc mask
with rasterio.open(LABEL_PATH) as src:
    mask_raw = src.read(1).astype(np.uint8)

# Debug mask values
print(f"Unique values trong mask: {np.unique(mask_raw)}")
print(f"Pixel > 127: {np.sum(mask_raw > 127)} / {mask_raw.size} ({np.sum(mask_raw > 127)/mask_raw.size*100:.1f}%)")
print(f"Pixel = 255: {np.sum(mask_raw == 255)}")
print(f"Pixel = 1  : {np.sum(mask_raw == 1)}")

# Tìm contours
_, binary = cv2.threshold(mask_raw, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Lọc và tạo bboxes
bboxes = []
for cnt in contours:
    if cv2.contourArea(cnt) < 100:
        continue
    bboxes.append(cv2.boundingRect(cnt))
print(f"Tìm được {len(bboxes)} bounding box")

# ── Panel 1: ảnh gốc ─────────────────────────────────────────
panel_orig = img_bgr.copy()

# ── Panel 2: ảnh gốc + bounding box ─────────────────────────
panel_bbox = img_bgr.copy()
for (x, y, w, h) in bboxes:
    cv2.rectangle(panel_bbox, (x, y), (x+w, y+h), (0, 255, 136), 2)
    cv2.putText(panel_bbox, "landslide",
                (x, max(y-6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 136), 1)

# ── Panel 3: mask ─────────────────────────────────────────────
panel_mask = cv2.cvtColor(mask_raw, cv2.COLOR_GRAY2BGR)

# ── Ghép 3 panel ngang ───────────────────────────────────────
# Thêm label tiêu đề cho mỗi panel
def add_title(img, title):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (512, 28), (30, 30, 30), -1)
    cv2.putText(out, title, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out

p1 = add_title(panel_orig, "1. Anh goc")
p2 = add_title(panel_mask, "2. Binary mask")
p3 = add_title(panel_bbox, f"3. Bounding box ({len(bboxes)})")

combined = np.hstack([p1, p2, p3])

cv2.imwrite(OUT, combined)
print(f"Saved: {OUT}")
os.startfile(OUT)