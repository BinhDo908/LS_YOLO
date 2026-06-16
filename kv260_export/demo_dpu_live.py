"""
demo_dpu_live.py — Chạy LS-YOLO (head Decoupled_Detect) trên KV260 DPU.
DPU : backbone + neck + toàn bộ conv của head.
CPU : decode_decoupled (sigmoid + grid) + NMS — vài ms.

Nguồn ảnh qua biến SOURCE (mặc định "0" = webcam):
  SOURCE=0                  -> camera USB (lưu ý: model train ảnh top-down,
                               webcam mặt đất sẽ cho box vô nghĩa — domain gap)
  SOURCE=/path/anh.jpg      -> 1 ảnh tĩnh (nên dùng để test domain khớp: ảnh aerial)
  SOURCE=/path/video.mp4    -> file video
  SOURCE=/path/folder       -> chạy lần lượt mọi ảnh trong folder
"""
import os, sys, time, glob, json
import numpy as np
import cv2
import torch
import vart, xir

CUR = os.path.dirname(os.path.abspath(__file__))
LSYOLO_SRC = os.environ.get("LSYOLO_SRC", os.path.join(CUR, ".."))
sys.path.insert(0, LSYOLO_SRC)
sys.path.insert(0, CUR)
from utils.general import non_max_suppression, scale_boxes
from ls_yolo_dpu import decode_decoupled

MODEL = os.environ.get("MODEL", os.path.join(CUR, "ls_yolo_landslide.xmodel"))
CONSTS = os.environ.get("CONSTS", os.path.join(os.path.dirname(MODEL), "head_constants.json"))
SOURCE = os.environ.get("SOURCE", "0")
CONF_THRES = float(os.environ.get("CONF", "0.25"))
IOU_THRES = float(os.environ.get("IOU", "0.45"))

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_consts():
    if os.path.exists(CONSTS):
        c = json.load(open(CONSTS))
    else:  # fallback = giá trị đã verify cho Improve.yaml
        c = {"nc": 1, "na": 3, "nl": 3, "stride": [8.0, 16.0, 32.0],
             "anchors": [[[1.25, 1.625], [2.0, 3.75], [4.125, 2.875]],
                         [[1.875, 3.8125], [3.875, 2.8125], [3.6875, 7.4375]],
                         [[3.625, 2.8125], [4.875, 6.1875], [11.65625, 10.1875]]]}
        print("[WARN] Không thấy head_constants.json, dùng fallback đã verify.")
    c["anchors"] = torch.tensor(c["anchors"]).float()
    return c


def get_dpu_runner():
    graph = xir.Graph.deserialize(MODEL)
    sgs = [s for s in graph.get_root_subgraph().toposort_child_subgraph()
           if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
    if not sgs:
        raise RuntimeError("Không tìm thấy DPU subgraph trong xmodel!")
    print(f"[INFO] {len(sgs)} DPU subgraph (1 là tốt nhất; nhiều = có op rớt CPU)")
    return vart.Runner.create_runner(sgs[0], "run")


def frames_from_source():
    """Yield (frame_bgr). Hỗ trợ camera / video / ảnh / folder."""
    if SOURCE.isdigit():
        cap = cv2.VideoCapture(int(SOURCE))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError("Không mở được camera!")
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield fr
        cap.release()
    elif os.path.isdir(SOURCE):
        for f in sorted(glob.glob(os.path.join(SOURCE, "*"))):
            if f.lower().endswith(IMG_EXT):
                im = cv2.imread(f)
                if im is not None:
                    yield im
    elif SOURCE.lower().endswith(IMG_EXT):
        im = cv2.imread(SOURCE)
        if im is None:
            raise RuntimeError(f"Không đọc được ảnh: {SOURCE}")
        while True:           # giữ cửa sổ để xem
            yield im.copy()
    else:                     # video
        cap = cv2.VideoCapture(SOURCE)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield fr
        cap.release()


def main():
    c = load_consts()
    nc, na, stride, anchors = c["nc"], c["na"], c["stride"], c["anchors"]

    runner = get_dpu_runner()
    it = runner.get_input_tensors()
    ot = runner.get_output_tensors()
    _, h, w, _ = it[0].dims
    in_scale = 2 ** it[0].get_attr("fix_point")
    out_scales = [2 ** -t.get_attr("fix_point") for t in ot]
    idata = [np.empty(tuple(it[0].dims), dtype=np.int8, order="C")]
    odata = [np.empty(tuple(t.dims), dtype=np.int8, order="C") for t in ot]
    torch.set_num_threads(4)

    print(f"[INFO] Input {w}x{h}, {len(ot)} output. SOURCE={SOURCE}. Nhấn Q để thoát.")
    for frame in frames_from_source():
        t0 = time.time()
        img = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2RGB)
        idata[0][0, ...] = (img.astype(np.float32) / 255.0 * in_scale).astype(np.int8)
        jid = runner.execute_async(idata, odata)
        runner.wait(jid)
        t_dpu = (time.time() - t0) * 1000

        t1 = time.time()
        # NHWC int8 -> NCHW float, rồi gom theo grid size về đúng thứ tự level (stride 8/16/32)
        outs = []
        for k, o in enumerate(odata):
            f = o.astype(np.float32) * np.float32(out_scales[k])
            outs.append(torch.from_numpy(f.transpose(0, 3, 1, 2).copy()))
        outs.sort(key=lambda t: t.shape[2], reverse=True)   # H lớn nhất = level 0 (stride 8)

        preds = decode_decoupled(outs, anchors, stride, nc, na)
        boxes = non_max_suppression(preds, conf_thres=CONF_THRES, iou_thres=IOU_THRES)[0]
        t_cpu = (time.time() - t1) * 1000

        if boxes is not None and len(boxes):
            boxes[:, :4] = scale_boxes((h, w), boxes[:, :4], frame.shape).round()
            for *xy, conf, cls in boxes:
                x1, y1, x2, y2 = map(int, xy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"landslide {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        fps = 1000.0 / (t_dpu + t_cpu + 1e-6)
        cv2.putText(frame, f"FPS:{fps:.1f} DPU:{t_dpu:.1f}ms CPU:{t_cpu:.1f}ms",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("LS-YOLO KV260", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
