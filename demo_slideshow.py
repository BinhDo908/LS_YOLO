import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import tifffile

IMG_DIR = r"D:\Training\Landslide_Dataset\Tài liệu ching chong\Moxi town（UAV-0.2m）\img"
WEIGHTS = "runs/train/ls_yolo_v22/weights/best.onnx"
OUT     = Path("runs/demo_video/moxi_detected.mp4")
FPS     = 30
IMGSZ   = 512
CONF    = 0.25
IOU     = 0.45
COLOR   = (0, 0, 255)


# ── Inference helpers (same as demo.py) ──────────────────────────────────────

def letterbox(im, new_shape=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top  = int(round(dh - 0.1)); bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1)); right  = int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def preprocess(frame_bgr, imgsz):
    img, ratio, (dw, dh) = letterbox(frame_bgr, new_shape=imgsz)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return img, ratio, dw, dh


def nms(boxes, scores, iou_thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        inter = np.maximum(0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]])) * \
                np.maximum(0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]))
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thresh]
    return keep


def postprocess(pred, ratio, dw, dh, orig_h, orig_w):
    pred  = pred[0]
    score = pred[:, 4] * pred[:, 5]
    mask  = score >= CONF
    pred  = pred[mask]; score = score[mask]
    if len(pred) == 0:
        return []
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2], axis=1)
    keep  = nms(boxes, score, IOU)
    boxes = boxes[keep]; score = score[keep]
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - dw) / ratio, 0, orig_w)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - dh) / ratio, 0, orig_h)
    return list(zip(boxes.astype(int), score.tolist()))


def draw(frame, detections):
    for (x1, y1, x2, y2), conf in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)
        label = f"landslide {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), COLOR, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


def normalize_to_uint8(arr):
    if arr.dtype == np.uint8:
        return arr
    p2, p98 = np.percentile(arr, 2), np.percentile(arr, 98)
    return np.clip((arr - p2) / (p98 - p2 + 1e-6) * 255, 0, 255).astype(np.uint8)


def read_tif(path):
    data = tifffile.imread(str(path))
    if data.ndim == 2:
        data = np.stack([data] * 3, axis=-1)
    elif data.ndim == 3 and data.shape[0] <= 4 and data.shape[0] < data.shape[1]:
        data = np.transpose(data, (1, 2, 0))
    rgb = normalize_to_uint8(data[:, :, :3])
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    img_dir = Path(IMG_DIR)
    paths = sorted(img_dir.glob("*.tif"))
    if not paths:
        paths = sorted(img_dir.glob("*.tiff"))
    print(f"Found {len(paths)} images in {img_dir.name}")

    # Read first image to get frame size
    sample = read_tif(paths[0])
    h, w = sample.shape[:2]
    print(f"Frame size: {w}x{h}  |  FPS: {FPS}  |  Duration: {len(paths)/FPS:.1f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))

    session  = ort.InferenceSession(WEIGHTS, providers=["CPUExecutionProvider"])
    in_name  = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    print("Processing...")
    t0 = time.perf_counter()
    for i, p in enumerate(paths):
        frame = read_tif(p)
        inp, ratio, dw, dh = preprocess(frame, IMGSZ)
        pred = session.run([out_name], {in_name: inp})[0]
        dets = postprocess(pred, ratio, dw, dh, h, w)
        frame = draw(frame, dets)
        writer.write(frame)

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {i+1}/{len(paths)}  |  {elapsed:.0f}s  |  {len(dets)} det(s) on last frame")

    writer.release()
    elapsed = time.perf_counter() - t0
    print(f"Done. {len(paths)} frames in {elapsed:.1f}s")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
