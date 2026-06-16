"""Audit chat luong label + ve bbox len 1 anh. CPU-only, khong dung GPU."""
import os, random
import cv2
import numpy as np
from pathlib import Path

SPLIT = os.environ.get("LS_SPLIT", "datasets/splits/train_uav_only.txt")
LABEL_DIR = os.environ.get("LS_LABELS", "datasets/labels_yolo_clean")
OUT = os.environ.get("LS_OUT", "datasets/label_check.png")
random.seed(7)


def img_to_label(img_path):
    p = Path(img_path)
    return Path(LABEL_DIR) / p.parent.parent.name / (p.stem + ".txt")


def read_tif(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            from PIL import Image
            img = np.array(Image.open(str(path)))
        except Exception:
            return None
    return img


def to_uint8_bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] >= 3:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        mn, mx = float(img.min()), float(img.max())
        img = ((img - mn) / (mx - mn) * 255).astype(np.uint8) if mx > mn else np.zeros_like(img, np.uint8)
    return img


lines = [l.strip() for l in open(SPLIT, encoding='utf-8', errors='ignore') if l.strip()]
random.shuffle(lines)

print("=== AUDIT 15 label (UAV) ===")
print(f"{'#':<3}{'boxes':<7}{'avg_area%':<11}{'min_area%':<11}location")
audited = []
for ip in lines:
    lp = img_to_label(ip)
    if not lp.exists():
        continue
    boxes = []
    for ln in open(lp):
        pr = ln.split()
        if len(pr) == 5:
            boxes.append(tuple(map(float, pr[1:])))
    audited.append((ip, lp, boxes))
    if len(audited) <= 15:
        areas = [w * h * 100 for _, _, w, h in boxes]
        loc = Path(ip).parent.parent.name
        print(f"{len(audited):<3}{len(boxes):<7}{(np.mean(areas) if areas else 0):<11.3f}{(min(areas) if areas else 0):<11.3f}{loc}")
    if len(audited) >= 40:
        break

# thong ke tong
allb = [b for _, _, bs in audited for b in bs]
nbg = sum(1 for _, _, bs in audited if not bs)
print(f"\nTong {len(audited)} anh: {nbg} background (rong), trung binh {len(allb)/max(1,len(audited)):.1f} box/anh")
if allb:
    aa = np.array([w * h * 100 for _, _, w, h in allb])
    print(f"Box area%: min {aa.min():.3f} | median {np.median(aa):.3f} | max {aa.max():.2f}")
    full = sum(1 for a in aa if a > 90)
    print(f"Box gan full-anh (>90% dien tich): {full}/{len(allb)}")

# ve 1 anh co 2..15 box
cand = [a for a in audited if 2 <= len(a[2]) <= 15] or [a for a in audited if a[2]]
ip, lp, boxes = cand[0]
img = to_uint8_bgr(read_tif(ip))
H, W = img.shape[:2]
th = max(2, int(max(H, W) / 400))
for cx, cy, w, h in boxes:
    x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
    x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), th)
scale = 1000 / max(H, W)
if scale < 1:
    img = cv2.resize(img, (int(W * scale), int(H * scale)))
cv2.imwrite(OUT, img)
print(f"\nDa ve: {Path(ip).name} ({W}x{H}, {len(boxes)} box) -> {OUT}")
print(f"Label: {lp}")
