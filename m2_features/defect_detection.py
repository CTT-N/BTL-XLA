"""
PHÁT HIỆN KHUYẾT TẬT

Các bước:
1. Làm mịn Gaussian nhẹ
  2. Co mặt nạ quả để bỏ viền (viền hay tối do bóng)
  3. Xác định MÀU LÀNH tham chiếu:
       K-means K=2 trên màu pixel (Ch.6). Nếu 2 cụm tách xa nhau (có vùng hỏng rộng) -> chọn cụm
       "lành" = cụm có nhiều pixel nằm trong dải màu đạt của loại quả (Color Slicing, Ch.3);
       nếu không thì dùng trung vị toàn quả.
  4. Màu lành CỤC BỘ tại từng pixel = trung bình Gaussian có trọng số (loại pixel nghi hỏng, lặp 3 vòng)
     => bóng đổ mượt (rìa tối hơn tâm) KHÔNG bị nhận nhầm là vết; vết thâm sắc nét thì nổi lên.
  5. Khoảng cách Euclid RGB:  D(z,a) = sqrt((zR-aR)^2+(zG-aG)^2+(zB-aB)^2) > D0
     Ngưỡng cục bộ = max(Otsu(D), D0_min)
     + pixel lệch xa màu lành toàn quả (d_global_min) + pixel quá tối (V < dark_v).
     Điểm chói phản quang (S thấp, V cao) bị loại trừ.
  6. Opening xoá đốm li ti, Closing lấp lỗ
  7. Connected Component Labeling: đếm vết + diện tích từng vết
  8. P_loi = sum(A_benh) / A_qua * 100%
"""
import cv2
import numpy as np
import config
from .color_features import ripe_pixel_mask


def _ellipse(k):
    k = max(1, int(k)) | 1  # ép về số lẻ
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _ripe_fraction(pix_bgr, fruit_cfg):
    """Tỉ lệ pixel (N,3 BGR uint8) nằm trong dải màu đạt."""
    hsv = cv2.cvtColor(pix_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return float(ripe_pixel_mask(hsv[:, 0], hsv[:, 1], hsv[:, 2], fruit_cfg).mean())


def estimate_healthy_reference(pix_bgr, fruit_cfg, sep_min, max_samples=4000):
    """
    Ước lượng màu "lành" của cả quả bằng K-means + Color Slicing
    Trả về (ref_healthy_bgr, split, defect_center_bgr | None)
        split=True  : phát hiện quả có 2 vùng màu khác biệt (vd vỏ đỏ + vùng thối nâu)
        split=False : chỉ 1 vùng màu (khác nhau chỉ do bóng đổ) -> ref = trung vị toàn quả
    Quy tắc tách:
        - cụm nhỏ nhất < 5% mẫu                      -> không tách
        - cả 2 cụm đều >= 80% pixel màu đạt          -> không tách (vd táo 2 màu đỏ-xanh)
        - chênh lệch tỉ lệ màu đạt giữa 2 cụm > 0.3  -> tách; cụm "lành" = cụm có nhiều màu đạt hơn
        - ngược lại: tách nếu 2 tâm cụm cách nhau >= sep_min; cụm lành = cụm lớn hơn
    """
    pix = pix_bgr.astype(np.float32)
    med = np.median(pix, axis=0)
    if len(pix) < 200:
        return med, False, None

    rng = np.random.default_rng(0)
    sub = pix[rng.choice(len(pix), min(max_samples, len(pix)), replace=False)]
    cv2.setRNGSeed(0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, cen = cv2.kmeans(sub, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lab = lab.ravel()
    sizes = [int((lab == k).sum()) for k in (0, 1)]
    if min(sizes) < 0.05 * len(sub):
        return med, False, None

    members = [sub[lab == k] for k in (0, 1)]
    if fruit_cfg is not None:
        rf = [_ripe_fraction(m, fruit_cfg) for m in members]
        if min(rf) >= 0.8:
            return med, False, None
        if abs(rf[0] - rf[1]) > 0.3:
            h = int(np.argmax(rf))
        elif np.linalg.norm(cen[0] - cen[1]) >= sep_min:
            h = int(np.argmax(sizes))
        else:
            return med, False, None
    else:
        if np.linalg.norm(cen[0] - cen[1]) < sep_min:
            return med, False, None
        h = int(np.argmax(sizes))
    return np.median(members[h], axis=0), True, np.median(members[1 - h], axis=0)

def _fit_radial_anomaly(hsv, mask, inside, fit_mask, cfg, fruit_cfg=None):
    """
    Khớp quy luật hình cầu từ tâm (rho=0) ra viền (rho=1) dựa trên S và V:
    - Saturation tăng ra viền: S_exp = a_s + b_s * (rho^power)  (b_s >= 0)
    - Value giảm ra viền:      V_exp = a_v + b_v * (rho^power)  (b_v <= 0)
    Chỉ đẩy vào `cluster_def` khi thỏa mãn cả điều kiện suy giảm S/V VÀ nằm trong dải Hue nghi lỗi của loại quả đó.
    """
    f_cfg = fruit_cfg or {}
    power = cfg.get("radial_sat_power", 2.0)
    sat_thr = cfg.get("radial_sat_thr", 28.0)
    val_thr = cfg.get("radial_val_thr", 20.0)
    sat_max_cap = f_cfg.get("cluster_sat_max", cfg.get("cluster_sat_max", 175.0))
    hue_ranges = f_cfg.get("cluster_hue_ranges", cfg.get("cluster_hue_ranges", [(0, 14)]))

    dist_to_edge = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_d = float(dist_to_edge.max())
    if max_d < 1.0 or fit_mask.sum() < 50:
        return np.zeros(mask.shape, dtype=bool)

    rho = np.clip(1.0 - (dist_to_edge / max_d), 0.0, 1.0)
    x_feat = rho ** power

    h_u8 = hsv[..., 0]
    s_f = hsv[..., 1].astype(np.float32)
    v_f = hsv[..., 2].astype(np.float32)

    # 1. Fit Saturation trên vùng fit_mask
    b_s, a_s = np.polyfit(x_feat[fit_mask], s_f[fit_mask], 1)
    if b_s < 0:
        a_s, b_s = float(np.median(s_f[fit_mask])), 0.0
    s_exp = a_s + b_s * x_feat

    # 2. Fit Value bền vững
    v_vals = v_f[fit_mask]
    bright_enough = fit_mask & (v_f >= np.percentile(v_vals, 25))
    v_fit_mask = bright_enough if bright_enough.sum() >= 50 else fit_mask

    b_v, a_v = np.polyfit(x_feat[v_fit_mask], v_f[v_fit_mask], 1)
    if b_v > 0:
        a_v, b_v = float(np.median(v_f[v_fit_mask])), 0.0
    v_exp = a_v + b_v * x_feat

    # Điều kiện 1: Saturation tụt thấp hơn đường cong kỳ vọng
    cond_sat_drop = (s_exp - s_f) > sat_thr
    # Điều kiện 2: Vùng tối hơn kỳ vọng NHƯNG Saturation phải thấp hơn ngưỡng trần
    cond_dark_dull = (v_f < (v_exp - val_thr)) & (s_f < sat_max_cap)

    # Điều kiện 3: Hue nằm trong dải màu vết hỏng của loại quả tương ứng
    cond_hue = np.zeros(h_u8.shape, dtype=bool)
    for lo, hi in hue_ranges:
        cond_hue |= (h_u8 >= lo) & (h_u8 <= hi)

    return (cond_sat_drop | cond_dark_dull) & cond_hue & inside

def estimate_local_reference(img_f32, inside, sigma, init_w, d_thr, fallback, n_iter=3):
    """
    Màu lành cục bộ (h,w,3) bằng normalized Gaussian convolution trong vùng quả.
    init_w: trọng số khởi tạo (loại sẵn vùng hỏng rộng). Mỗi vòng loại thêm pixel có D > d_thr.
    """
    w = init_w.astype(np.float32)
    ref = np.tile(fallback.astype(np.float32), (*inside.shape, 1))
    for it in range(n_iter):
        num = cv2.GaussianBlur(img_f32 * w[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(w, (0, 0), sigma)
        ok = den > 1e-3
        ref = np.where(ok[..., None], num / np.maximum(den, 1e-3)[..., None], fallback[None, None, :])
        if it < n_iter - 1:
            d = np.sqrt(((img_f32 - ref) ** 2).sum(axis=2))
            w_new = (inside & (d < d_thr)).astype(np.float32)
            if w_new.sum() < 50:
                break
            w = w_new
    return ref.astype(np.float32)


def detect_defects(crop_bgr, mask, cfg=None, fruit_cfg=None, debug=False):
    cfg = cfg or config.DEFECT
    fruit_area = int(cv2.countNonZero(mask))
    empty = dict(defect_mask=np.zeros_like(mask), defect_ratio=0.0, n_spots=0,
                 defect_area_px=0, spots=[], ref_bgr=(0, 0, 0), threshold=0.0,
                 dist_map=np.zeros(mask.shape, np.float32))
    if debug:
        empty["debug_steps"] = {}
    if fruit_area == 0:
        return empty

    # 1) làm mịn nhẹ
    k = int(cfg["smooth_k"]) | 1
    smooth = cv2.GaussianBlur(crop_bgr, (k, k), 0)
    smooth_f = smooth.astype(np.float32)

    # 2) co mặt nạ, bỏ viền
    d_e = 2.0 * np.sqrt(fruit_area / np.pi)
    e = max(int(cfg["erode_px"]), int(round(cfg["erode_ratio"] * d_e)))
    inner = cv2.erode(mask, _ellipse(2 * e + 1), borderType=cv2.BORDER_CONSTANT, borderValue=0) if e > 0 else mask.copy()
    if cv2.countNonZero(inner) < 50:
        inner = mask.copy()
    inside = inner > 0

    # 3) Phân cụm Lành / Lỗi theo quy luật giảm Saturation bậc 2 của hình cầu
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    hl_raw = ((hsv[..., 1] < cfg["highlight_s_max"]) & (v > cfg["highlight_v_min"])).astype(np.uint8) * 255
    highlight = cv2.dilate(hl_raw, _ellipse(5), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0

    fit_mask = inside & ~highlight
    if fit_mask.sum() < 50:
        fit_mask = inside

    # Vòng 1: Fit sơ bộ trên toàn bộ vùng không chói
    cluster_def = _fit_radial_anomaly(hsv, mask, inside, fit_mask, cfg, fruit_cfg) & ~highlight

    # Vòng 2 (Refit): Fit lại đường cong hình cầu trên vùng SẠCH (đã bỏ cluster_def)
    clean_fit = fit_mask & ~cluster_def
    if clean_fit.sum() >= 50:
        cluster_def = _fit_radial_anomaly(hsv, mask, inside, clean_fit, cfg, fruit_cfg) & ~highlight

    # Suy ra màu lành đại diện `ref` và khoảng cách toàn cục `dist_g` từ cụm lành còn lại
    healthy_pixels = smooth_f[clean_fit if clean_fit.sum() >= 50 else fit_mask]
    ref = np.median(healthy_pixels, axis=0).astype(np.float32)
    dist_g = np.sqrt(((smooth_f - ref[None, None, :]) ** 2).sum(axis=2))
    split = bool(cluster_def.any())

    # 4) màu lành cục bộ (bù bóng đổ - loại bỏ cả cụm hỏng lẫn điểm chói khi nội suy)
    sigma = max(5.0, cfg["ref_sigma_ratio"] * d_e)
    init_w = (fit_mask & ~cluster_def) if split else fit_mask
    if init_w.sum() < 50:
        init_w = inside
    ref_map = estimate_local_reference(smooth_f, inside, sigma, init_w, cfg["d0_min"], ref)

    # 5) khoảng cách màu + ngưỡng
    dist = np.sqrt(((smooth_f - ref_map) ** 2).sum(axis=2))
    thr = float(cfg["d0_min"])
    if cfg["use_otsu"]:
        d8 = np.clip(dist[fit_mask], 0, 255).astype(np.uint8).reshape(-1, 1)
        otsu_t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(thr, float(otsu_t))

    cand = (dist > thr) & inside
    if dist_g is not None:
        cand |= (dist_g > cfg["d_global_min"]) & inside
    cand |= cluster_def
    cand |= (v < cfg["dark_v"]) & inside
    cand &= ~highlight
    raw = cand.astype(np.uint8) * 255

    # 6) Morphology: Opening rồi Closing
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, _ellipse(cfg["open_k"]), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, _ellipse(cfg["close_k"]), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    raw = cv2.bitwise_and(raw, mask)

    # 7) Connected Component Labeling
    n_lab, labels, stats, cents = cv2.connectedComponentsWithStats(raw, connectivity=8)
    min_area = max(cfg["min_spot_area_px"], cfg["min_spot_area_ratio"] * fruit_area)
    defect_mask = np.zeros_like(mask)
    spots = []
    for i in range(1, n_lab):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        defect_mask[labels == i] = 255
        spots.append(dict(area=a,
                          centroid=(float(cents[i][0]), float(cents[i][1])),
                          bbox=(int(stats[i, 0]), int(stats[i, 1]),
                                int(stats[i, 2]), int(stats[i, 3]))))

    # 8) tỉ lệ lỗi
    defect_area = int(sum(s["area"] for s in spots))
    res = dict(defect_mask=defect_mask,
               defect_ratio=100.0 * defect_area / fruit_area,
               n_spots=len(spots), defect_area_px=defect_area, spots=spots,
               ref_bgr=tuple(float(x) for x in ref), threshold=thr, dist_map=dist)
    if debug:
        res["debug_steps"] = _build_defect_debug(
            crop_bgr, mask, inner, inside, ref, cluster_def, ref_map, dist, raw, defect_mask, spots
        )
    return res

# phục vụ ảnh tĩnh trên web
def _build_defect_debug(crop_bgr, mask, inner, inside, ref, cluster_def, ref_map, dist, raw, defect_mask, spots):
    bg = (32, 36, 48)
    base = crop_bgr.copy()
    base[mask == 0] = bg

    # 2.3.1: Viền mask gốc (xanh) & viền inner sau khi gọt (vàng)
    erode_vis = base.copy()
    for cnt_mask, color in ((mask, (0, 255, 0)), (inner, (0, 255, 255))):
        cnts, _ = cv2.findContours(cnt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(erode_vis, cnts, -1, color, 2)

    # 2.3.2: Ảnh phân cụm (Xanh ngọc = Cụm lành, Đỏ = Cụm nghi hỏng)
    kmeans_vis = base.copy()
    healthy_zone = inside & ~cluster_def
    kmeans_vis[healthy_zone] = (kmeans_vis[healthy_zone] * 0.4 + np.array([54, 188, 148]) * 0.6).astype(np.uint8)
    kmeans_vis[cluster_def] = (kmeans_vis[cluster_def] * 0.35 + np.array([24, 72, 255]) * 0.65).astype(np.uint8)

    # 2.3.3: Bản đồ màu lành cục bộ đã bù bóng đổ
    ref_map_vis = np.clip(ref_map, 0, 255).astype(np.uint8)
    ref_map_vis[~inside] = bg

    # 2.3.4: Heatmap khoảng cách cục bộ sau khi trừ bóng đổ
    d_u8 = np.clip(dist * 2.5, 0, 255).astype(np.uint8) * inside.astype(np.uint8)
    dist_local_vis = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)

    # 2.3.6: Ảnh tổng hợp cuối cùng + khung bbox từng vết
    overlay = base.copy()
    overlay[defect_mask > 0] = (0, 0, 255)
    final_vis = cv2.addWeighted(base, 0.65, overlay, 0.35, 0)
    for idx, s in enumerate(spots, 1):
        x, y, w_box, h_box = s["bbox"]
        cv2.rectangle(final_vis, (x, y), (x + w_box, y + h_box), (0, 255, 255), 1)
        cv2.putText(final_vis, f"#{idx} ({s['area']}px)", (x, max(12, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return {
        "2_erode_border_vis": erode_vis,
        "3_kmeans_cluster_vis": kmeans_vis,
        "4_local_ref_map_bgr": ref_map_vis,
        "5_dist_local_heatmap": dist_local_vis,
        "6_morphology_cleaned_mask": raw.copy(),
        "7_final_overlay_bgr": final_vis,
    }