"""
sample_eval.py — Lấy 1000 cặp có landslide từ folder UAV.
Chạy: python3 sample_eval.py
"""
import os, cv2, random, shutil, numpy as np
from pathlib import Path

DATASET = os.environ.get("LS_IMAGES", "datasets/images")
OUT_IMG = os.environ.get("LS_TEST_IMAGES", "datasets/test_images")
OUT_LBL = os.environ.get("LS_TEST_LABELS", "datasets/test_labels")
TOTAL   = 1000
SIZE    = 512
FILTER  = "UAV"   # chỉ lấy folder có chữ UAV

Path(OUT_IMG).mkdir(parents=True, exist_ok=True)
Path(OUT_LBL).mkdir(parents=True, exist_ok=True)
random.seed(42)

# Thu thập cặp (img, mask) từ folder UAV có landslide
all_pairs = []
for event in sorted(Path(DATASET).iterdir()):
    if not event.is_dir(): continue
    if FILTER not in event.name: continue   # chỉ lấy folder UAV
    img_dir  = event / "img"
    mask_dir = event / "mask"
    if not img_dir.exists() or not mask_dir.exists(): continue

    for img_path in sorted(img_dir.glob("*.tif")):
        if img_path.name.startswith('.'): continue
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            for ext in ['.tif', '.tiff', '.png']:
                alt = mask_dir / (img_path.stem + ext)
                if alt.exists(): mask_path = alt; break
        if not mask_path.exists(): continue
        all_pairs.append((img_path, mask_path))

print(f"Tổng cặp UAV tìm thấy: {len(all_pairs)}")
random.shuffle(all_pairs)

# Chuyển ảnh .tif -> .jpg
def to_jpg(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            from PIL import Image
            img = np.array(Image.open(str(path)))
        except: return None
    if img is None: return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3:
        if img.shape[2] == 4: img = img[:,:,:3]
        elif img.shape[2] > 4: img = img[:,:,:3]
    if img.dtype != np.uint8:
        mn, mx = img.min(), img.max()
        img = ((img-mn)/(mx-mn+1e-5)*255).astype(np.uint8) if mx > mn else np.zeros_like(img, dtype=np.uint8)
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)

# Chuyển mask .tif -> YOLO bbox
def to_yolo(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        try:
            from PIL import Image
            mask = np.array(Image.open(str(mask_path)))
        except: return []
    if mask is None: return []
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask[:,:,:3], cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mn, mx = mask.min(), mask.max()
        mask = ((mask-mn)/(mx-mn+1e-5)*255).astype(np.uint8) if mx > mn else np.zeros_like(mask, dtype=np.uint8)
    # mask dùng 0/1 -> ngưỡng 0
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    if np.sum(binary > 0) < 5: return []

    num_labels, labels_im = cv2.connectedComponents(binary)
    H, W = binary.shape
    boxes = []
    for lbl in range(1, num_labels):
        ys, xs = np.where(labels_im == lbl)
        if len(xs) < 10: continue
        cx = ((xs.min()+xs.max())/2) / W
        cy = ((ys.min()+ys.max())/2) / H
        bw = (xs.max()-xs.min()) / W
        bh = (ys.max()-ys.min()) / H
        if bw*bh < 0.0005 or bw*bh > 0.98: continue
        boxes.append((cx, cy, bw, bh))
    return boxes

# Lưu ảnh và label
saved = 0
for img_path, mask_path in all_pairs:
    if saved >= TOTAL: break

    boxes = to_yolo(mask_path)
    if not boxes: continue

    img = to_jpg(img_path)
    if img is None: continue

    stem = (f"{img_path.parent.parent.name}_{img_path.stem}"
            .replace(" ", "_")
            .replace("(", "").replace(")", "")
            .replace("（", "").replace("）", ""))

    cv2.imwrite(os.path.join(OUT_IMG, f"{stem}.jpg"), img,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(os.path.join(OUT_LBL, f"{stem}.txt"), "w") as f:
        f.write("\n".join([f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                           for cx, cy, bw, bh in boxes]))
    saved += 1
    if saved % 100 == 0:
        print(f"  [{saved}/{TOTAL}]...")

print(f"\n[OK] Đã lưu {saved} cặp có landslide từ folder UAV")
print(f"  images -> {OUT_IMG}")
print(f"  labels -> {OUT_LBL}")