"""
MODULE 1 - TÁCH TỪNG QUẢ & VẼ BIÊN (Chương 5)
    extract_objects  : Connected Component Labeling -> List[FruitObject]
    extract_boundary : beta(A) = A - (A (-) B)  (biên hình thái dày 1 pixel)
"""
import cv2
import numpy as np
import config
from common.contracts import FruitObject


def extract_objects(frame_bgr, mask_full, min_area=None):
    """
    Input : frame BGR gốc, mask_full uint8 {0,255}
    Output: List[FruitObject] sắp xếp theo x tăng dần
    """
    min_area = config.SEGMENT["min_area"] if min_area is None else min_area
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_full, connectivity=8)
    objs = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                   cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        comp = ((labels[y:y + h, x:x + w] == i).astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        objs.append(FruitObject(bbox=(x, y, w, h),
                                crop_bgr=frame_bgr[y:y + h, x:x + w].copy(), mask=comp,
                                contour=max(cnts, key=cv2.contourArea)))
    objs.sort(key=lambda o: o.bbox[0])
    return objs


def extract_boundary(mask, k=3):
    """Biên hình thái: beta(A) = A - (A erode B). Input mask {0,255}; Output ảnh biên {0,255}."""
    # borderValue=0: coi ngoài crop là nền, nếu không viền sẽ bị hở tại các điểm chạm mép crop
    eroded = cv2.erode(mask, np.ones((k, k), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.subtract(mask, eroded)


