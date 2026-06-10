import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from pathlib import Path
import numpy as np
import torch
import cv2
import rasterio

# ── Config ────────────────────────────────────────────────────────────────────
WEIGHTS   = "runs/train/ls_yolo_v1/weights/best.pt"
IMGSZ     = 512
CONF_THRES = 0.35   # detection confidence threshold
IOU_THRES  = 0.45   # NMS IoU threshold
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"

# Default test image (change this to any .tif you want)
DEFAULT_IMAGE = r"D:\Training\Landslide_Dataset\Tài liệu ching chong\palu\img\Palu0513.tif"

# One representative image per source type for quick cross-dataset testing
SAMPLE_IMAGES = [
    (r"D:\Training\Landslide_Dataset\Tài liệu ching chong\Jiuzhai valley (UAV-0.2m)\img\jiuzzhaigou_0.2m_UVA0500.tif",  "UAV 0.2m"),
    (r"D:\Training\Landslide_Dataset\Tài liệu ching chong\Jiuzhai valley (UAV-0.5m)\img\jiuzzhaigou_0.5m_UVA0500.tif",  "UAV 0.5m"),
    (r"D:\Training\Landslide_Dataset\Tài liệu ching chong\Longxi River（UAV）\img\longxihe_UAV0500.tif",                  "UAV Longxi"),
    (r"D:\Training\Landslide_Dataset\Tài liệu ching chong\Moxi town（UAV-0.2m）\img\moxizheng_0.2m_UAV0500.tif",         "UAV Moxi 0.2m"),
    (r"D:\Training\Landslide_Dataset\Tài liệu ching chong\Moxitaidi (UAV-0.6m)\img\luding_UAV0500.tif",                  "UAV Moxi 0.6m"),
    (r"D:\Training\Landslide_Dataset\Tài liệu ching chong\Longxi River(SAT)\img\LongxiheSAT0500.tif",                    "SAT Longxi"),
    (r"D:\Training\Landslide_Dataset\Tài liệu ching chong\Tiburon Peninsula（Sentinel）\img\Haiti002.tif",                "SAT Sentinel"),
    (r"D:\Training\Landslide_Dataset\Tài liệu ching chong\Tiburon Peninsula（planet）\img\Haiti_Planet001.tif",           "SAT Planet"),
    (r"D:\Training\Landslide_Dataset\Tài liệu ching chong\palu\img\Palu0513.tif",                                        "SAT Palu"),
    (r"D:\Training\Landslide_Dataset\Tài liệu ching chong\Moxi town（UAV-0.2m）\img\moxizheng_0.2m_UAV0430.tif",         "UAV Moxi 0.2m"),
    (r"D:\Training\Landslide_Dataset\Tài liệu ching chong\Hokkaido Iburi-Tobu\img\Hokkaido1136.tif", "UAV Hokkaido"),
    (r"D:\Training\Landslide_Dataset\Tài liệu ching chong\Hokkaido Iburi-Tobu\img\Hokkaido1138.tif", "SAT Hokkaido"),
]

# ── Load model ────────────────────────────────────────────────────────────────
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

device = select_device(DEVICE)
model  = DetectMultiBackend(WEIGHTS, device=device, fp16=False)
model.eval()
stride, names = model.stride, model.names
print(f"Model loaded: {WEIGHTS}")
print(f"Device: {device}  |  Classes: {names}")

# ── Inference function ────────────────────────────────────────────────────────
def detect_image(img_path: str):
    """Run detection on a single TIF (or any image) and return (img_rgb, detections)."""
    img_path = Path(img_path)

    # Read image (TIF or standard format)
    print(f"  [1] Reading image...")
    if img_path.suffix.lower() in ('.tif', '.tiff'):
        with rasterio.open(str(img_path)) as src:
            data = src.read()          # (bands, H, W)
        # Use first 3 bands as RGB; clip to uint8
        rgb = np.stack([data[i] for i in range(min(3, data.shape[0]))], axis=-1)
        if rgb.dtype != np.uint8:
            p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
            rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)
    else:
        import cv2
        bgr = cv2.imread(str(img_path))
        rgb = bgr[:, :, ::-1]
    print(f"  [1] Image shape: {rgb.shape}, dtype: {rgb.dtype}")

    # Letterbox resize to model input size
    print(f"  [2] Letterbox resize...")
    img_lb, ratio, pad = letterbox(rgb, IMGSZ, stride=int(stride), auto=True)
    img_tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float().to(device) / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # (1,3,H,W)
    print(f"  [2] Tensor shape: {img_tensor.shape}")

    # Forward pass
    print(f"  [3] Forward pass...")
    with torch.no_grad():
        preds = model(img_tensor)
    print(f"  [3] Done.")

    # NMS
    print(f"  [4] NMS (conf={CONF_THRES})...")
    preds = non_max_suppression(preds, CONF_THRES, IOU_THRES, max_det=300)
    det = preds[0]  # (N, 6) = xyxy + conf + cls
    print(f"  [4] {len(det)} raw detections after NMS.")

    # Scale boxes back to original image size
    detections = []
    if len(det):
        det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], rgb.shape).round()
        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = [int(v.item()) for v in xyxy]
            detections.append({
                'bbox':  [x1, y1, x2, y2],
                'conf':  float(conf),
                'class': int(cls),
                'label': names[int(cls)],
            })

    return rgb, detections


def visualize(img_path: str, save_dir: str = "runs/infer"):
    """Detect and save result image as PNG using OpenCV."""
    rgb, detections = detect_image(img_path)
    H, W = rgb.shape[:2]

    # Convert RGB → BGR for OpenCV drawing
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        label = f"{d['label']} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        ty = max(y1 - 5, th + 2)
        cv2.rectangle(canvas, (x1, ty - th - 2), (x1 + tw, ty + 2), (0, 0, 255), -1)
        cv2.putText(canvas, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if not detections:
        msg = "No landslide detected"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(canvas, msg, ((W - tw) // 2, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)

    # Save PNG
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(img_path).stem + "_result.png")
    cv2.imwrite(str(out_path), canvas)

    # Print results
    print(f"\n{'-'*50}")
    print(f"Image  : {Path(img_path).name}  ({W}×{H})")
    print(f"Found  : {len(detections)} detection(s)")
    for i, d in enumerate(detections, 1):
        x1, y1, x2, y2 = d['bbox']
        cx = (x1+x2)/2/W;  cy = (y1+y2)/2/H
        bw = (x2-x1)/W;    bh = (y2-y1)/H
        print(f"  [{i}] {d['label']}  conf={d['conf']:.3f}  "
              f"bbox=({cx:.3f}, {cy:.3f}, {bw:.3f}, {bh:.3f})")
    print(f"\n  Saved -> {out_path.resolve()}")
    print(f"{'-'*50}\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # User passed explicit paths / folders
        targets = []
        for t in sys.argv[1:]:
            p = Path(t)
            if p.is_dir():
                targets.extend([(str(f), f.stem) for f in sorted(p.glob("**/*.tif"))[:5]])
            else:
                targets.append((str(p), p.stem))
    else:
        # Default: one sample per source type
        targets = [(p, label) for p, label in SAMPLE_IMAGES if Path(p).exists()]
        missing = [(p, label) for p, label in SAMPLE_IMAGES if not Path(p).exists()]
        if missing:
            print(f"Skipping {len(missing)} missing files:")
            for p, label in missing:
                print(f"  [{label}] {Path(p).name}")

    print(f"\nRunning inference on {len(targets)} image(s)...\n")
    for img_path, label in targets:
        print(f"[{label}] {Path(img_path).name}")
        try:
            visualize(img_path)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
