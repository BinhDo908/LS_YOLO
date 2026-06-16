"""
Test inference on a single image.
Edit IMAGE_PATH to point to your image, then Run (F5 or Ctrl+F5).
Result is saved to runs/infer/ and shown in a popup window.
"""

import sys, os
# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from pathlib import Path
import numpy as np
import torch
import cv2

# CONFIG — only edit these 3 lines
IMAGE_PATH = r"C:\Users\Admin\Pictures\test.tif"
WEIGHTS    = "runs/train/ls_yolo_v22/weights/best.pt"
CONF_THRES = 0.35    # confidence threshold (0-1)

IOU_THRES  = 0.45
IMGSZ      = 512
DEVICE     = "cuda:0" if torch.cuda.is_available() else "cpu"
SAVE_DIR   = Path("runs/infer")

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    return np.clip((arr - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)


def read_image(img_path: Path) -> np.ndarray:
    """Read TIF or standard image, return RGB uint8 numpy array."""
    if img_path.suffix.lower() in ('.tif', '.tiff'):
        # Try tifffile first (best for multi-band GeoTIFF)
        try:
            import tifffile
            data = tifffile.imread(str(img_path))
            if data.ndim == 2:
                data = np.stack([data] * 3, axis=-1)
            elif data.ndim == 3 and data.shape[0] <= 4 and data.shape[0] < data.shape[1]:
                data = np.transpose(data, (1, 2, 0))   # (C,H,W) -> (H,W,C)
            rgb = data[:, :, :3]
            return _normalize_to_uint8(rgb)
        except Exception:
            pass

        # Fallback: PIL/Pillow
        try:
            from PIL import Image
            img = Image.open(str(img_path))
            data = np.array(img)
            if data.ndim == 2:
                data = np.stack([data] * 3, axis=-1)
            elif data.ndim == 3 and data.shape[2] > 3:
                data = data[:, :, :3]
            return _normalize_to_uint8(data)
        except Exception:
            pass

        # Last fallback: OpenCV
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(
                f"Cannot read TIF: {img_path}\n"
                "Install tifffile:  pip install tifffile"
            )
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run(img_path: str):
    img_path = Path(img_path)
    if not img_path.exists():
        print(f"[ERROR] File not found: {img_path}")
        return

    # Load model
    print(f"[1/4] Loading model: {WEIGHTS}")
    device = select_device(DEVICE)
    model  = DetectMultiBackend(WEIGHTS, device=device, fp16=False)
    model.eval()
    stride, names = model.stride, model.names
    print(f"      Device={device}  Classes={names}")

    # Read image
    print(f"[2/4] Reading image: {img_path.name}")
    rgb = read_image(img_path)
    H, W = rgb.shape[:2]
    print(f"      Size: {W}x{H}")

    # Preprocess
    img_lb, _, _ = letterbox(rgb, IMGSZ, stride=int(stride), auto=True)
    tensor = torch.from_numpy(img_lb).permute(2, 0, 1).float().to(device) / 255.0
    tensor = tensor.unsqueeze(0)

    # Inference
    print(f"[3/4] Inference (conf>={CONF_THRES})...")
    with torch.no_grad():
        preds = model(tensor)
    preds = non_max_suppression(preds, CONF_THRES, IOU_THRES, max_det=300)
    det   = preds[0]

    # Scale boxes back to original size
    detections = []
    if len(det):
        det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], rgb.shape).round()
        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = [int(v.item()) for v in xyxy]
            detections.append({"bbox": [x1, y1, x2, y2],
                                "conf": float(conf),
                                "label": names[int(cls)]})

    # Draw results
    print(f"[4/4] Drawing results -- {len(detections)} detection(s)")
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{d['label']} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        ty = max(y1 - 5, th + 2)
        cv2.rectangle(canvas, (x1, ty - th - 2), (x1 + tw, ty + 2), (0, 0, 255), -1)
        cv2.putText(canvas, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if not detections:
        msg = "No landslide detected"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.putText(canvas, msg, ((W - tw) // 2, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 0), 2)

    # Save result
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAVE_DIR / f"{img_path.stem}_result.png"
    cv2.imwrite(str(out_path), canvas)

    # Print summary
    print(f"\n{'─'*55}")
    print(f"  Image  : {img_path.name}  ({W}x{H})")
    print(f"  Weights: {WEIGHTS}")
    print(f"  Result : {len(detections)} landslide detection(s)")
    for i, d in enumerate(detections, 1):
        x1, y1, x2, y2 = d["bbox"]
        cx = (x1+x2)/2/W;  cy = (y1+y2)/2/H
        bw = (x2-x1)/W;    bh = (y2-y1)/H
        print(f"    [{i}] conf={d['conf']:.3f}  "
              f"center=({cx:.3f},{cy:.3f})  size=({bw:.3f}x{bh:.3f})")
    print(f"  Saved  : {out_path.resolve()}")
    print(f"{'─'*55}\n")

    # Show popup
    try:
        disp = cv2.resize(canvas, (min(W, 1280), min(H, 720)))
        cv2.imshow(f"LS-YOLO | {img_path.name}", disp)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        print("(Cannot show popup -- see saved file)")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    run(target)
