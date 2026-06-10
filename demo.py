import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

WEIGHTS = "runs/train/ls_yolo_v22/weights/best.onnx"
IMGSZ   = 512
CONF    = 0.25
IOU     = 0.45
CLASSES = ["landslide"]
COLOR   = (0, 0, 255)  # BGR red


def letterbox(im, new_shape=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right  = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def preprocess(img_bgr, imgsz):
    img, ratio, (dw, dh) = letterbox(img_bgr, new_shape=imgsz)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]  # HWC → BCHW
    return img, ratio, dw, dh


def nms(boxes, scores, iou_thresh):
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    x2 = boxes[:, 2]; y2 = boxes[:, 3]
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
    pred = pred[0]  # (N, 6)
    obj_conf  = pred[:, 4]
    cls_conf  = pred[:, 5]
    score     = obj_conf * cls_conf
    mask      = score >= conf_thresh
    pred      = pred[mask]; score = score[mask]
    if len(pred) == 0:
        return []

    # xywh → xyxy (letterbox space)
    cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1 = cx - bw / 2; y1 = cy - bh / 2
    x2 = cx + bw / 2; y2 = cy + bh / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(boxes, score, iou_thresh)
    boxes = boxes[keep]; score = score[keep]

    # scale back to original image
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
    boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

    return list(zip(boxes.astype(int), score.tolist()))


def draw(img, detections):
    for (x1, y1, x2, y2), conf in detections:
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR, 2)
        label = f"landslide {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), COLOR, -1)
        cv2.putText(img, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img


def run(source, weights, imgsz, conf_thresh, iou_thresh, save, show):
    session = ort.InferenceSession(weights, providers=["CPUExecutionProvider"])
    in_name  = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    src = Path(source)
    if src.is_dir():
        paths = sorted(src.glob("*"))
        paths = [p for p in paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}]
    else:
        paths = [src]

    save_dir = Path("runs/demo")
    save_dir.mkdir(parents=True, exist_ok=True)

    for p in paths:
        img_bgr = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Cannot read {p}, skipping")
            continue

        orig_h, orig_w = img_bgr.shape[:2]
        inp, ratio, dw, dh = preprocess(img_bgr, imgsz)

        t0 = time.perf_counter()
        pred = session.run([out_name], {in_name: inp})[0]
        ms = (time.perf_counter() - t0) * 1000

        dets = postprocess(pred, ratio, dw, dh, orig_h, orig_w, conf_thresh, iou_thresh)

        print(f"{p.name}: {len(dets)} detection(s)  |  inference {ms:.1f} ms")

        result = draw(img_bgr.copy(), dets)

        if save:
            out_path = save_dir / p.name
            cv2.imwrite(str(out_path), result)
            print(f"  saved → {out_path}")

        if show:
            win = "LS-YOLO Demo  (any key = next, q = quit)"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.imshow(win, result)
            key = cv2.waitKey(0)
            if key == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    DEFAULT = r"C:\Users\Admin\Pictures\test.tif"
    parser.add_argument("source", nargs="?", default=DEFAULT, help="image file or folder")
    parser.add_argument("--weights",  default=WEIGHTS)
    parser.add_argument("--imgsz",    type=int,   default=IMGSZ)
    parser.add_argument("--conf",     type=float, default=CONF)
    parser.add_argument("--iou",      type=float, default=IOU)
    parser.add_argument("--save",     action="store_true", default=True,  help="save result image")
    parser.add_argument("--no-show",  action="store_true", help="skip display window")
    args = parser.parse_args()

    run(
        source     = args.source,
        weights    = args.weights,
        imgsz      = args.imgsz,
        conf_thresh= args.conf,
        iou_thresh = args.iou,
        save       = args.save,
        show       = not args.no_show,
    )
