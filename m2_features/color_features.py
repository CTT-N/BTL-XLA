"""
PHÂN TÍCH MÀU SẮC
- compute_color_histogram(): histogram Hue chuẩn hoá
- extract_color_features(): mean H/S/V + tỉ lệ màu đạt (bỏ qua vùng viền tối)
"""
import cv2
import numpy as np
import config
from common.utils import ellipse_kernel

"""
    Histogram kênh Hue (thang OpenCV 0..179) chỉ trên pixel thuộc quả, chuẩn hoá tổng = 1.
        p(h_k) = n_k / N   (Ch.2)
    Trả về: np.ndarray shape (bins,)
"""
def compute_color_histogram(crop_bgr, mask, bins=32):
    
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], mask, [bins], [0, 180]).flatten()
    total = hist.sum()
    return hist / total if total > 0 else hist


def _circular_mean_hue(h_vals):
    """Hue là góc (đỏ nằm ở 2 đầu 0 và 179) nên phải lấy trung bình vòng tròn."""
    ang = h_vals.astype(np.float64) * 2.0 * np.pi / 180.0
    m = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    return float((m * 180.0 / (2.0 * np.pi)) % 180.0)


# Color Slicing
def ripe_pixel_mask(h, s, v, fruit_cfg):
    in_range = np.zeros(h.shape, bool)
    for i, (lo, hi) in enumerate(fruit_cfg["ripe_hue_ranges"]):
        ls, hs = fruit_cfg["ripe_s"][i]
        lv, hv = fruit_cfg["ripe_v"][i]
        in_range |= (
            (h >= lo) & (h <= hi)
            & (s >= ls) & (s <= hs)
            & (v >= lv) & (v <= hv)
        )
    return in_range

"""
    Input : crop_bgr (h,w,3) uint8, mask (h,w) uint8 {0,255}, fruit = 'apple' | 'orange'
    Output: dict {mean_h, mean_s, mean_v, hist_h, ripe_ratio}
        ripe_ratio = tỉ lệ pixel của quả đạt màu chín (dải viền ngoài cùng mặc định tính đạt).
"""
def extract_color_features(crop_bgr, mask, fruit="apple", fruit_cfg=None, bins=None, cfg=None):
    fruit_cfg = fruit_cfg or config.FRUIT_CFG[fruit]
    bins = bins or config.COLOR["hist_bins"]
    cfg = cfg or config.DEFECT

    fruit_area = int(cv2.countNonZero(mask))
    if fruit_area == 0:
        return dict(mean_h=0.0, mean_s=0.0, mean_v=0.0,
                    hist_h=np.zeros(bins), ripe_ratio=0.0)

    # 1) Làm mịn nhẹ trước khi chuyển HSV
    k = int(cfg["smooth_k"]) | 1
    smooth = cv2.GaussianBlur(crop_bgr, (k, k), 0)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)

    # 2) Co mặt nạ tách phần ruột (inner) và dải viền (rim)
    d_e = 2.0 * np.sqrt(fruit_area / np.pi)
    e = max(int(cfg["erode_px"]), int(round(cfg["erode_ratio"] * d_e)))
    inner = cv2.erode(mask, ellipse_kernel(2 * e + 1), borderType=cv2.BORDER_CONSTANT, borderValue=0) if e > 0 else mask.copy()
    if cv2.countNonZero(inner) < 50:
        inner = mask.copy()

    full_inside = mask > 0
    core_inside = inner > 0
    rim_mask = full_inside & ~core_inside  # Dải viền ngoài cùng

    # 3) Phần ruột kiểm tra theo ngưỡng màu; dải viền mặc định đạt (True)
    h_core = hsv[..., 0][core_inside]
    s_core = hsv[..., 1][core_inside]
    v_core = hsv[..., 2][core_inside]

    ripe_core_count = int(ripe_pixel_mask(h_core, s_core, v_core, fruit_cfg).sum())
    rim_count = int(rim_mask.sum())
    total_count = int(full_inside.sum())

    return dict(
        mean_h=_circular_mean_hue(h_core),
        mean_s=float(s_core.mean()),
        mean_v=float(v_core.mean()),
        hist_h=compute_color_histogram(smooth, inner, bins),
        ripe_ratio=float((ripe_core_count + rim_count) / total_count),
    )