"""
Fix labels: remove full-image bboxes (cx~0.5, cy~0.5, w~1, h~1)
from labels_yolo_fixed -> labels_yolo_clean
"""
import os
import shutil
from pathlib import Path

SRC = Path(r"D:\Training\labels_yolo_fixed")
DST = Path(r"D:\Training\labels_yolo_clean")

def is_full_image(cx, cy, w, h):
    return abs(cx - 0.5) < 0.02 and abs(cy - 0.5) < 0.02 and w > 0.98 and h > 0.98

stats = {"total": 0, "fixed_mixed": 0, "cleared": 0, "unchanged": 0}

for src_file in SRC.rglob("*.txt"):
    rel = src_file.relative_to(SRC)
    dst_file = DST / rel
    dst_file.parent.mkdir(parents=True, exist_ok=True)

    lines = src_file.read_text(encoding="utf-8").splitlines()
    kept = []
    removed = 0

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            kept.append(line)
            continue
        try:
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            kept.append(line)
            continue
        if is_full_image(cx, cy, w, h):
            removed += 1
        else:
            kept.append(line)

    stats["total"] += 1
    if removed > 0 and len(kept) > 0:
        stats["fixed_mixed"] += 1
    elif removed > 0 and len(kept) == 0:
        stats["cleared"] += 1
    else:
        stats["unchanged"] += 1

    content = "\n".join(kept)
    if kept:
        content += "\n"
    dst_file.write_text(content, encoding="utf-8")

print(f"Tong file xu ly  : {stats['total']}")
print(f"Fixed (mixed)    : {stats['fixed_mixed']}  <- xoa bbox toan anh, giu bbox nho")
print(f"Cleared          : {stats['cleared']}  <- chi co bbox toan anh -> thanh background")
print(f"Unchanged        : {stats['unchanged']}")
print(f"\nLabel sach da luu vao: {DST}")
