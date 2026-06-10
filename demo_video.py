import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

URL     = "https://www.youtube.com/watch?v=8hjSUsf0AJU"
START   = "00:03:00"
END     = "00:05:25"
WEIGHTS = "runs/train/ls_yolo_v22/weights/best.onnx"
IMGSZ   = 512
CONF    = 0.25
IOU     = 0.45
COLOR   = (0, 0, 255)
OUT_DIR = Path("runs/demo_video")


# ── Preprocessing ────────────────────────────────────────────────────────────

def letterbox(im, new_shape=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top    = int(round(dh - 0.1)); bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1)); right  = int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def preprocess(frame_bgr, imgsz):
    img, ratio, (dw, dh) = letterbox(frame_bgr, new_shape=imgsz)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]
    return img, ratio, dw, dh


# ── NMS ──────────────────────────────────────────────────────────────────────

def nms(boxes, scores, iou_thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thresh]
    return keep


def postprocess(pred, ratio, dw, dh, orig_h, orig_w, conf_thresh, iou_thresh):
    pred = pred[0]
    score = pred[:, 4] * pred[:, 5]
    mask  = score >= conf_thresh
    pred  = pred[mask]; score = score[mask]
    if len(pred) == 0:
        return []
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2], axis=1)
    keep  = nms(boxes, score, iou_thresh)
    boxes = boxes[keep]; score = score[keep]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
    boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)
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


# ── Main ─────────────────────────────────────────────────────────────────────

def download_segment(url, start, end, out_path):
    print(f"Downloading {start} to {end} ...")
    cmd = [
        "yt-dlp",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", str(out_path),
        "--no-playlist",
        url,
    ]
    subprocess.run(cmd, check=True)
    print(f"Saved: {out_path}")



def process_video(video_path, out_path, weights, imgsz, conf, iou):
    session  = ort.InferenceSession(weights, providers=["CPUExecutionProvider"])
    in_name  = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    cap = cv2.VideoCapture(str(video_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    print(f"Processing {total} frames at {fps:.1f} fps ...")
    frame_idx = 0
    t0 = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        inp, ratio, dw, dh = preprocess(frame, imgsz)
        pred = session.run([out_name], {in_name: inp})[0]
        dets = postprocess(pred, ratio, dw, dh, height, width, conf, iou)
        frame = draw(frame, dets)
        writer.write(frame)

        if frame_idx % 30 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  frame {frame_idx}/{total}  |  {elapsed:.0f}s elapsed  |  {len(dets)} det(s)")

    cap.release()
    writer.release()
    elapsed = time.perf_counter() - t0
    print(f"Done. {frame_idx} frames in {elapsed:.1f}s -> {out_path}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_video = OUT_DIR / "segment_raw.mp4"
    out_video = OUT_DIR / "segment_detected.mp4"

    if not raw_video.exists():
        download_segment(URL, START, END, raw_video)

    process_video(raw_video, out_video, WEIGHTS, IMGSZ, CONF, IOU)
