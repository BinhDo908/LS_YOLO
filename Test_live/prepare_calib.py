"""
prepare_calib.py — Lấy 1000 ảnh CÓ landslide để calibrate quantization.
Chạy: python3 prepare_calib.py
"""
import os, shutil, random, cv2, numpy as np
from pathlib import Path

# CẤU HÌNH
DATASET_ROOT = os.environ.get("LS_IMAGES", "datasets/images")
OUTPUT       = os.environ.get("LS_CALIB", "datasets/calib_images")
TOTAL        = 1000

# Lọc theo tên folder — để "" nếu muốn lấy TẤT CẢ folder
# Ví dụ: FILTER = "UAV"  -> chỉ lấy folder có chữ "UAV"
#         FILTER = "SAT"  -> chỉ lấy folder có chữ "SAT"
#         FILTER = ""     -> lấy tất cả
FILTER = "UAV"

MIN_WHITE = 5   # pixel trắng tối thiểu trong mask

Path(OUTPUT).mkdir(parents=True, exist_ok=True)
random.seed(42)

# Tìm event folder
event_folders = []
for d in sorted(Path(DATASET_ROOT).iterdir()):
    if not d.is_dir(): continue
    if not (d / "img").exists(): continue
    if not (d / "mask").exists(): continue
    if FILTER and FILTER not in d.name: continue
    event_folders.append(d)

if not event_folders:
    print(f"[ERROR] Không tìm thấy folder nào (FILTER='{FILTER}')")
    exit(1)

print(f"[INFO] Tìm thấy {len(event_folders)} event folders (FILTER='{FILTER}'):")
for e in event_folders:
    print(f"  - {e.name}")

# Thu thập ảnh CÓ landslide
print("\n[INFO] Đang quét ảnh có landslide...")
all_valid = []

for event in event_folders:
    img_dir  = event / "img"
    mask_dir = event / "mask"
    count = 0

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in ('.tif', '.tiff', '.jpg', '.png'):
            continue
        if img_path.name.startswith('.'):
            continue

        # Tìm mask tương ứng
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            for ext in ('.tif', '.tiff', '.png'):
                alt = mask_dir / (img_path.stem + ext)
                if alt.exists():
                    mask_path = alt
                    break

        if not mask_path.exists():
            continue

        # Kiểm tra có landslide (mask dùng 0/1)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if np.sum(mask > 0) < MIN_WHITE:
            continue

        all_valid.append(img_path)
        count += 1

    print(f"  {event.name}: {count} ảnh có landslide")

print(f"\n[INFO] Tổng ảnh có landslide: {len(all_valid)}")

if not all_valid:
    print("[ERROR] Không tìm thấy ảnh nào!")
    exit(1)

# Lấy ngẫu nhiên TOTAL ảnh
random.shuffle(all_valid)
sampled = all_valid[:TOTAL]

print(f"[INFO] Đang copy {len(sampled)} ảnh vào:\n  {OUTPUT}")
copied = 0
for img_path in sampled:
    event_name = img_path.parent.parent.name
    out_name = (f"{event_name}_{img_path.name}"
                .replace(" ", "_")
                .replace("(", "").replace(")", "")
                .replace("（", "").replace("）", ""))
    shutil.copy(img_path, Path(OUTPUT) / out_name)
    copied += 1
    if copied % 100 == 0:
        print(f"  [{copied}/{len(sampled)}]...")

print(f"\n[OK] Đã copy {copied} ảnh CÓ landslide")
print(f"[INFO] Bước tiếp: copy folder calib_images vào ~/Downloads/KV260/ rồi chạy quantize_calib.py")
