"""
Debug script: mimics train.py's first batch with full error capture.
Run with: conda run -n ls-yolo python debug_train_step.py
"""
import sys
import traceback
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

device = torch.device('cuda:0')

# ── 1. Load hyp and data config ──────────────────────────────────────────────
print("\n[1] Loading configs...")
with open('data/hyps/hyp.scratch-low.yaml') as f:
    hyp = yaml.safe_load(f)

with open('data/landslide.yaml') as f:
    data_dict = yaml.safe_load(f)

import utils.dataloaders as _dl
_dl._LABEL_DIR = data_dict['label_dir']
_dl._IMG_SUBDIR = data_dict.get('img_subdir', 'img')
print("  Configs loaded.")

# ── 2. Build model exactly as train.py does ───────────────────────────────────
print("\n[2] Building model...")
try:
    from models.yolo import Model
    nc = data_dict['nc']
    model = Model('models/landslide/Improve.yaml', ch=3, nc=nc, anchors=None).to(device)
    model.hyp = hyp
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model built: {n_params:,} params")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── 3. Create fake batch ──────────────────────────────────────────────────────
print("\n[3] Creating fake batch (bs=16, 512x512)...")
batch_size = 16
imgsz = 512
imgs = torch.rand(batch_size, 3, imgsz, imgsz, device=device)

# targets: [img_idx, class, cx, cy, w, h]  (normalized 0-1)
n_tgt = 20
targets = torch.zeros(n_tgt, 6, device=device)
targets[:, 0] = torch.randint(0, batch_size, (n_tgt,)).float()
targets[:, 1] = 0
targets[:, 2:] = 0.5 + torch.randn(n_tgt, 4) * 0.1
targets[:, 2:].clamp_(0.05, 0.95)
print(f"  imgs: {imgs.shape}, targets: {targets.shape}")

# ── 4. Set up ComputeLoss ─────────────────────────────────────────────────────
print("\n[4] Creating ComputeLoss...")
try:
    from utils.loss import ComputeLoss
    compute_loss = ComputeLoss(model)
    print("  ComputeLoss created.")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── 5. AMP scaler ─────────────────────────────────────────────────────────────
amp_enabled = True
scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.937, weight_decay=5e-4)

# ── 6. Forward pass ───────────────────────────────────────────────────────────
print(f"\n[5] Forward pass (amp={amp_enabled})...")
try:
    with torch.amp.autocast('cuda', enabled=amp_enabled):
        pred = model(imgs)
    torch.cuda.synchronize()
    print("  Forward OK.")
    if isinstance(pred, (list, tuple)):
        for i, p in enumerate(pred):
            print(f"    pred[{i}].shape = {p.shape}")
    else:
        print(f"  pred shape = {pred.shape}")
except Exception as e:
    print(f"  FORWARD FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── 7. Loss ───────────────────────────────────────────────────────────────────
print("\n[6] Computing loss...")
try:
    with torch.amp.autocast('cuda', enabled=amp_enabled):
        loss, loss_items = compute_loss(pred, targets)
    torch.cuda.synchronize()
    print(f"  Loss OK: {loss.item():.4f}  items={loss_items.tolist()}")
except Exception as e:
    print(f"  LOSS FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── 8. Backward ───────────────────────────────────────────────────────────────
print("\n[7] Backward pass...")
try:
    scaler.scale(loss).backward()
    torch.cuda.synchronize()
    print("  Backward OK.")
except Exception as e:
    print(f"  BACKWARD FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── 9. Optimizer step ─────────────────────────────────────────────────────────
print("\n[8] Optimizer step...")
try:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    print("  Optimizer step OK.")
except Exception as e:
    print(f"  OPTIMIZER FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== ALL STEPS PASSED — training step works correctly ===")
print(f"Peak VRAM used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
