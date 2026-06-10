from pathlib import Path

DATASET_ROOT = "D:\\Training\\Landslide_Dataset\\Tài liệu ching chong"
total = 0
for loc in Path(DATASET_ROOT).iterdir():
    if not loc.is_dir():
        continue
    img_dir = loc / "img"
    if not img_dir.exists():
        continue
    imgs = list(img_dir.glob("*.tif")) + list(img_dir.glob("*.TIF"))
    print(f"{loc.name}: {len(imgs)}")
    total += len(imgs)
print(f"\nTổng: {total} ảnh")