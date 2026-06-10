import numpy as np
import rasterio
import cv2

IMG_PATH   = r"D:\Training\Landslide_Dataset\Tài liệu ching chong\Hokkaido Iburi-Tobu\img\Hokkaido0042.tif"
LABEL_PATH = r"D:\Training\labels_yolo_fixed\Hokkaido Iburi-Tobu\Hokkaido0042.txt"

with rasterio.open(IMG_PATH) as src:
    img = np.stack([src.read(i) for i in range(1, src.count+1)], axis=-1)

# Normalize về uint8
img_show = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
img_bgr  = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)
H, W     = img_bgr.shape[:2]

with open(LABEL_PATH) as f:
    for line in f:
        cls, xc, yc, w, h = map(float, line.strip().split())
        x1 = int((xc - w/2) * W)
        y1 = int((yc - h/2) * H)
        x2 = int((xc + w/2) * W)
        y2 = int((yc + h/2) * H)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Verify label", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()