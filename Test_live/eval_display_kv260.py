"""
eval_display_kv260.py
Hiển thị realtime + tính P/R/mAP trên KV260.
Fix: compute_metrics sort theo confidence trước khi match GT.

Đặt tại  : ~/project/LS-YOLO/eval_display_kv260.py
Cần có   : test_images/  (ảnh .jpg)
           test_labels/  (nhãn .txt YOLO format)
Chạy     : python3 eval_display_kv260.py
Điều khiển: Q = thoát sớm (vẫn tính metric từ ảnh đã chạy)
"""
import os, sys, time
import cv2, numpy as np, torch
import vart, xir
from pathlib import Path

# CẤU HÌNH
MODEL      = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "ls_yolo_landslide.xmodel")
IMG_DIR    = os.environ.get("IMG_DIR", "/workspace/test_images")
LBL_DIR    = os.environ.get("LBL_DIR", "/workspace/test_labels")
DELAY_MS   = 1000
CONF_THRES = 0.25
IOU_THRES  = 0.45
NC, NA     = 1, 3
STRIDES    = [8, 16, 32]
ANCHORS    = [
    [10,13,  16,30,  33,23],
    [30,61,  62,45,  59,119],
    [116,90, 156,198, 373,326],
]


# Decode
def make_grid(nx, ny):
    yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing='ij')
    return torch.stack((xv, yv), 2).view(1,1,ny,nx,2).float()

def decode(raws):
    z = []
    for i, raw in enumerate(raws):
        bs, _, H, W = raw.shape
        r = raw[:, :4*NA       ].view(bs,NA,4, H,W).permute(0,1,3,4,2)
        c = raw[:, 4*NA:5*NA   ].view(bs,NA,1, H,W).permute(0,1,3,4,2)
        l = raw[:, 5*NA:       ].view(bs,NA,NC,H,W).permute(0,1,3,4,2)
        y = torch.cat([r,c,l], -1).sigmoid()
        g = make_grid(W, H)
        a = torch.tensor(ANCHORS[i]).float().view(1,NA,1,1,2)
        y[...,0:2] = (y[...,0:2]*2 - 0.5 + g) * STRIDES[i]
        y[...,2:4] = (y[...,2:4]*2)**2 * a
        z.append(y.view(bs,-1,5+NC))
    return torch.cat(z, 1)

def xywh2xyxy(x):
    y = x.clone()
    y[...,0] = x[...,0] - x[...,2]/2
    y[...,1] = x[...,1] - x[...,3]/2
    y[...,2] = x[...,0] + x[...,2]/2
    y[...,3] = x[...,1] + x[...,3]/2
    return y

def box_iou(b1, b2):
    def area(b): return (b[:,2]-b[:,0]) * (b[:,3]-b[:,1])
    inter = (torch.min(b1[:,None,2:], b2[:,2:]) -
             torch.max(b1[:,None,:2], b2[:,:2])).clamp(0).prod(2)
    return inter / (area(b1)[:,None] + area(b2) - inter + 1e-8)

def nms(pred):
    out = []
    for x in pred:
        x = x[x[:,4] > CONF_THRES]
        if not len(x): out.append(torch.zeros((0,6))); continue
        x[:,5:] *= x[:,4:5]
        box = xywh2xyxy(x[:,:4])
        conf, cls = x[:,5:].max(1, keepdim=True)
        x = torch.cat((box, conf, cls.float()), 1)
        x = x[x[:,4] > CONF_THRES]
        if not len(x): out.append(torch.zeros((0,6))); continue
        keep, idxs = [], x[:,4].argsort(descending=True)
        while idxs.numel():
            i = idxs[0]; keep.append(i)
            if idxs.numel() == 1: break
            iou = box_iou(x[i:i+1,:4], x[idxs[1:],:4])[0]
            idxs = idxs[1:][iou <= IOU_THRES]
        out.append(x[torch.stack(keep)])
    return out

def scale_boxes(src, boxes, dst):
    g = min(src[0]/dst[0], src[1]/dst[1])
    px = (src[1] - dst[1]*g) / 2
    py = (src[0] - dst[0]*g) / 2
    boxes[:,[0,2]] -= px; boxes[:,[1,3]] -= py
    boxes[:,:4] /= g
    boxes[:,0].clamp_(0, dst[1]); boxes[:,1].clamp_(0, dst[0])
    boxes[:,2].clamp_(0, dst[1]); boxes[:,3].clamp_(0, dst[0])
    return boxes


# mAP
def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size-1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1]))

def compute_metrics(all_preds, all_labels, iou_thresh=0.5):
    """
    Tính P/R/AP chuẩn COCO:
    - Với mỗi ảnh, match predictions -> GT theo thứ tự confidence GIẢM DẦN
    - Một GT box chỉ được match 1 lần (used flag)
    - Sau đó sort toàn bộ detections theo confidence để tính P/R curve
    """
    tp_list, conf_list, n_gt = [], [], 0

    for preds, labels in zip(all_preds, all_labels):
        n_gt += len(labels)

        if not len(preds):
            continue

        if not len(labels):
            tp_list.extend([0] * len(preds))
            conf_list.extend(preds[:,4].tolist())
            continue

        # Sort predictions theo confidence giảm dần
        order = preds[:,4].argsort(descending=True)
        preds_s = preds[order]

        gt   = labels[:,1:].float()   # [M, 4] pixel xyxy
        det  = preds_s[:,:4].float()  # [N, 4]
        iou  = box_iou(det, gt)       # [N, M]
        used = torch.zeros(len(labels), dtype=torch.bool)

        for di in range(len(preds_s)):
            best_iou, best_j = iou[di].max(0)
            if best_iou >= iou_thresh and not used[best_j]:
                tp_list.append(1)
                used[best_j] = True
            else:
                tp_list.append(0)
            conf_list.append(preds_s[di, 4].item())

    if not conf_list:
        return 0.0, 0.0, 0.0

    # Sort toàn bộ theo confidence để vẽ P/R curve
    order  = np.argsort(conf_list)[::-1]
    tp_arr = np.array(tp_list)[order]
    cum_tp = np.cumsum(tp_arr)
    cum_fp = np.cumsum(1 - tp_arr)
    prec   = cum_tp / (cum_tp + cum_fp + 1e-8)
    rec    = cum_tp / (n_gt + 1e-8)
    ap     = compute_ap(rec, prec)
    return float(prec[-1]), float(rec[-1]), ap


# Main
def main():
    # Khởi DPU
    graph = xir.Graph.deserialize(MODEL)
    sg    = [c for c in graph.get_root_subgraph().toposort_child_subgraph()
             if c.has_attr("device") and c.get_attr("device").upper() == "DPU"]
    if not sg:
        print("[ERROR] Không có DPU subgraph!"); sys.exit(1)

    runner  = vart.Runner.create_runner(sg[0], "run")
    it, ot  = runner.get_input_tensors(), runner.get_output_tensors()
    _, h, w, _ = it[0].dims
    in_sc   = 2 ** it[0].get_attr("fix_point")
    out_scs = [2 ** -t.get_attr("fix_point") for t in ot]
    idata   = [np.empty(tuple(it[0].dims), dtype=np.int8, order="C")]
    odata   = [np.empty(tuple(t.dims),     dtype=np.int8, order="C") for t in ot]

    print(f"[INFO] DPU input : {tuple(it[0].dims)}")
    for k, t in enumerate(ot):
        print(f"[INFO] DPU output[{k}]: {tuple(t.dims)}")

    # Lấy danh sách ảnh có label
    exts  = ('.jpg','.jpeg','.png')
    imgs  = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(exts) and not f.startswith(".")])
    valid = [(f, Path(f).stem+".txt") for f in imgs
             if os.path.exists(os.path.join(LBL_DIR, Path(f).stem+".txt"))]

    print(f"[INFO] {len(valid)} ảnh có label -> bắt đầu eval")
    print("[INFO] Nhấn Q để dừng sớm (vẫn tính metric từ ảnh đã chạy)\n")

    cv2.namedWindow("LS-YOLO Eval", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("LS-YOLO Eval", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    all_preds, all_labels = [], []
    times_dpu, times_cpu  = [], []
    stopped_at = len(valid)

    for idx, (img_f, lbl_f) in enumerate(valid):
        frame = cv2.imread(os.path.join(IMG_DIR, img_f))
        if frame is None: continue
        oh, ow = frame.shape[:2]

        # Đọc label -> pixel xyxy
        gt_boxes = []
        lpath = os.path.join(LBL_DIR, lbl_f)
        for line in open(lpath, errors="ignore").read().strip().splitlines():
            parts = line.split()
            if len(parts) < 5: continue
            cls, cx, cy, bw, bh = map(float, parts[:5])
            x1=(cx-bw/2)*ow; y1=(cy-bh/2)*oh
            x2=(cx+bw/2)*ow; y2=(cy+bh/2)*oh
            gt_boxes.append([cls, x1, y1, x2, y2])
        gt_t = torch.tensor(gt_boxes) if gt_boxes else torch.zeros((0,5))

        # DPU inference
        t0  = time.time()
        img = cv2.cvtColor(cv2.resize(frame,(w,h)), cv2.COLOR_BGR2RGB)
        idata[0][0,...] = (img.astype(np.float32)/255.0*in_sc).astype(np.int8)
        jid = runner.execute_async(idata, odata); runner.wait(jid)
        t_dpu = (time.time()-t0)*1000

        # Decode + NMS
        t1 = time.time()
        raws = sorted(
            [torch.from_numpy((o.astype(np.float32)*np.float32(out_scs[k]))
                              .transpose(0,3,1,2).copy())
             for k,o in enumerate(odata)],
            key=lambda t: t.shape[2], reverse=True)
        preds = nms(decode(raws))[0]
        if len(preds):
            preds[:,:4] = scale_boxes((h,w), preds[:,:4], (oh,ow,3)).round()
        t_cpu = (time.time()-t1)*1000

        times_dpu.append(t_dpu)
        times_cpu.append(t_cpu)
        all_preds.append(preds)
        all_labels.append(gt_t)

        # Hiển thị
        disp    = frame.copy()
        n_det   = 0
        n_gt_img = len(gt_t)

        # Ground truth — xanh lá
        for row in gt_t:
            x1,y1,x2,y2 = map(int, row[1:])
            cv2.rectangle(disp,(x1,y1),(x2,y2),(0,255,0),2)

        # Prediction — đỏ
        if len(preds):
            n_det = len(preds)
            for *xy, conf, cls in preds:
                x1,y1,x2,y2 = map(int,xy)
                cv2.rectangle(disp,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(disp, f"landslide {conf:.2f}",
                            (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,(0,0,255),2)

        fps = 1000/(t_dpu+t_cpu+1e-6)
        cv2.rectangle(disp,(0,0),(ow,75),(0,0,0),-1)
        cv2.putText(disp, f"[{idx+1}/{len(valid)}] {img_f}",
                    (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2)
        cv2.putText(disp,
                    f"FPS:{fps:.1f}  DPU:{t_dpu:.0f}ms  CPU:{t_cpu:.0f}ms  "
                    f"GT:{n_gt_img}  Det:{n_det}",
                    (10,52), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2)
        cv2.putText(disp,"GT",   (ow-120,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.putText(disp,"Pred", (ow-70, 22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        print(f"[{idx+1:4d}/{len(valid)}] {img_f:40s} "
              f"DPU:{t_dpu:5.1f}ms CPU:{t_cpu:4.1f}ms "
              f"GT:{n_gt_img} Det:{n_det}")

        cv2.imshow("LS-YOLO Eval", disp)
        key = cv2.waitKey(DELAY_MS) & 0xFF
        if key == ord('q'):
            stopped_at = idx+1
            print(f"\n[INFO] Dừng sớm tại ảnh {stopped_at}")
            break

    cv2.destroyAllWindows()

    # Tính metrics
    print("\n[INFO] Đang tính metrics...")
    p50, r50, ap50 = compute_metrics(all_preds, all_labels, iou_thresh=0.5)

    ap_list = []
    for thr in np.arange(0.5, 1.0, 0.05):
        _, _, ap = compute_metrics(all_preds, all_labels, iou_thresh=float(thr))
        ap_list.append(ap)
    map5095 = float(np.mean(ap_list))

    t_dpu_mean = float(np.mean(times_dpu))
    t_cpu_mean = float(np.mean(times_cpu))
    fps_mean   = 1000 / (t_dpu_mean + t_cpu_mean)

    print("\n" + "="*55)
    print("  KẾT QUẢ EVAL TRÊN KV260  (DPU INT8)")
    print("="*55)
    print(f"  Số ảnh đã test    : {stopped_at}")
    print(f"  Conf threshold    : {CONF_THRES}")
    print(f"  IoU  threshold    : {IOU_THRES}")
    print("-"*55)
    print(f"  Precision (P)     : {p50:.4f}   ({p50*100:.2f}%)")
    print(f"  Recall    (R)     : {r50:.4f}   ({r50*100:.2f}%)")
    print(f"  mAP@0.5           : {ap50:.4f}   ({ap50*100:.2f}%)")
    print(f"  mAP@0.5:0.95      : {map5095:.4f}   ({map5095*100:.2f}%)")
    print("-"*55)
    print(f"  DPU trung bình    : {t_dpu_mean:.1f} ms")
    print(f"  CPU trung bình    : {t_cpu_mean:.1f} ms")
    print(f"  FPS trung bình    : {fps_mean:.1f}")
    print("="*55)


if __name__ == "__main__":
    main()
