"""
dryrun_export.py — DE-RISK TRƯỚC KHI TRAIN.
Dựng model từ cfg với WEIGHT NGẪU NHIÊN (không cần best.pt, không cần train),
quantize + xuất .xmodel. Mục đích DUY NHẤT: để compile thử bằng vai_c_xir và
ĐỌC SỐ DPU SUBGRAPH trong log.

  - 1 DPU subgraph        -> kiến trúc map trọn vẹn lên DPU, train thoải mái.
  - Nhiều subgraph        -> có op rớt CPU (nghi ngờ CAM dilation 3/5). Xem op
                             nào trong log, rồi chỉnh (vd dilation 3/5 -> 2/4).

Việc partitioning KHÔNG phụ thuộc weight, nên model random cho ra cùng kết quả
chia subgraph như model đã train. Chạy trong Docker Vitis-AI.
"""
import os, sys
import torch

LSYOLO_SRC = os.environ.get("LSYOLO_SRC", "/workspace/LS-YOLO")
sys.path.insert(0, LSYOLO_SRC)
sys.path.insert(0, os.path.join(LSYOLO_SRC, "kv260_export"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pytorch_nndct.apis import torch_quantizer
from models.yolo import Model
from ls_yolo_dpu import DecoupledDPU

CFG = os.environ.get("CFG", os.path.join(LSYOLO_SRC, "models/landslide/Improve.yaml"))
OUT = os.environ.get("OUT", "/workspace/compiled_dryrun")
IMG = 416   # khớp imgsz lúc train (phương án B); chia hết 32
os.makedirs(OUT, exist_ok=True)


def main():
    print(f"[INFO] Dựng model random từ: {CFG}")
    model = Model(CFG).float().eval()
    net = DecoupledDPU(model).eval()
    inp = torch.randn(1, 3, IMG, IMG)

    with torch.no_grad():
        out = net(inp)
    print(f"[CHECK] {len(out)} output: {[tuple(o.shape) for o in out]}")

    # calib nhanh bằng vài tensor random (chất lượng không quan trọng cho dry-run)
    q = torch_quantizer("calib", net, (inp,), output_dir=OUT)
    qm = q.quant_model
    with torch.no_grad():
        for _ in range(8):
            _ = qm(torch.randn(1, 3, IMG, IMG))
    q.export_quant_config()

    q2 = torch_quantizer("test", net, (inp,), output_dir=OUT)
    with torch.no_grad():
        _ = q2.quant_model(inp)
    q2.export_xmodel(deploy_check=False, output_dir=OUT)
    print("[DONE] xmodel dry-run tại:", OUT)
    print(">> Bước tiếp: compile rồi ĐỌC LOG xem '... DPU subgraph'.")


if __name__ == "__main__":
    main()
