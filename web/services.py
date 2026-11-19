import cv2
import numpy as np

import config
from m1_segmentation.pipeline_m1 import detect_fruits
from m2_features.pipeline_md2 import extract_all_features
from m3_classify_viz.classifier import classify_rule_based, load_thresholds

from pathlib import Path
import sys

WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
UPLOAD_DIR = WEB_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _encode_image(image):
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Không thể mã hoá ảnh")
    return "data:image/png;base64," + __import__("base64").b64encode(buf.tobytes()).decode("ascii")


def _overlay_mask(frame, mask, color=(54, 188, 148)):
    overlay = frame.copy()
    overlay[mask > 0] = (
        overlay[mask > 0] * 0.55 + np.array(color, dtype=np.float32) * 0.45
    ).astype(np.uint8)
    return overlay


def _draw_boundary(frame, object_data):
    out = frame.copy()
    x, y, w, h = object_data.bbox
    roi = out[y:y + h, x:x + w]
    mask = object_data.mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(roi, contours, -1, (35, 220, 170), 2)
    return out

def _dataset_images(fruit):
    test_dir = DATA_DIR / "test" / fruit
    if not test_dir.exists():
        return []
    return sorted(test_dir.glob("*/*.png"))



def analyze_image(image_bgr, fruit):
    original = image_bgr.copy()

    # 1. Pipeline M1
    debug_out = detect_fruits(original, return_debug=True)
    objs, mask, preprocessed, raw_mask = debug_out
    if not objs:
        return {"ok": False, "error": "Không phát hiện được quả trong ảnh.", "steps": []}
    obj = max(objs, key=lambda o: cv2.countNonZero(o.mask))

    # 2. Pipeline M2
    features, m2_masks = extract_all_features(obj, fruit, return_debug=True)

    # 3. Pipeline M3
    label, reason = classify_rule_based(features, load_thresholds(fruit))

    # --- Chỉ còn phần vẽ overlay hiển thị lên Web ---
    x, y, w, h = obj.bbox
    raw_mask_vis = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
    segmented_overlay = _overlay_mask(original, mask)

    # 2.1 Ảnh màu chín
    ripe_crop = obj.crop_bgr.copy()
    ripe_crop[m2_masks["ripe_mask"] > 0] = (ripe_crop[m2_masks["ripe_mask"] > 0] * 0.45 + np.array([54, 188, 148]) * 0.55).astype(np.uint8)
    ripe_crop[obj.mask == 0] = (32, 36, 48)
    ripe_overlay = np.full(original.shape, (32, 36, 48), dtype=np.uint8)
    ripe_overlay[y:y + h, x:x + w] = ripe_crop

    # 2.2 Ảnh vết hỏng
    defect_overlay = original.copy()
    full_dm = np.zeros(original.shape[:2], dtype=np.uint8)
    full_dm[y:y + h, x:x + w] = m2_masks["defect_mask"]
    defect_overlay[full_dm > 0] = (24, 72, 255)
    contours_def, _ = cv2.findContours(full_dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(defect_overlay, contours_def, -1, (0, 230, 255), 2)

    # 2.3 Ảnh hình dạng
    shape_vis = original.copy()
    cv2.drawContours(shape_vis, [obj.contour + np.array([[[x, y]]])], -1, (35, 220, 170), 2)
    cv2.rectangle(shape_vis, (x, y), (x + w, y + h), (255, 200, 0), 1)
    M = cv2.moments(obj.mask)
    if M["m00"] > 0:
        cv2.circle(shape_vis, (int(x + M["m10"] / M["m00"]), int(y + M["m01"] / M["m00"])), 4, (0, 0, 255), -1)

    # 3. Ảnh kết quả
    grade_colors = {"LOAI 1": (54, 188, 148), "LOAI 2": (48, 170, 255), "LOAI 3": (72, 72, 255)}
    result = original.copy()
    cv2.rectangle(result, (x, y), (x + w, y + h), grade_colors.get(label, (255, 255, 255)), 3)
    cv2.putText(result, label, (x, max(32, y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, grade_colors.get(label, (255, 255, 255)), 2, cv2.LINE_AA)
    
    
    steps = [
        {
            "number": "01",
            "title": "Tiền xử lý & Tách nền",
            "subtitle": "Lọc nhiễu · Phân ngưỡng · Morphology",
            "substep": [
                {
                    "title": "1.1 Ảnh gốc",
                    "description": "Ảnh trái cây dạng BGR được đọc từ dataset hoặc tệp tải lên.",
                    "img": _encode_image(original),
                    "metrics": [{"label": "Kích thước", "value": f"{original.shape[1]} x {original.shape[0]}"}],
                },
                {
                    "title": "1.2 Tiền xử lý",
                    "description": "Median blur, Gaussian blur và chỉnh sáng Gamma/HSV để giảm nhiễu trước khi tách nền.",
                    "img": _encode_image(preprocessed),
                    "metrics": [
                        {"label": "Median", "value": f"{config.PREPROCESS['median_k']} px"},
                        {"label": "Gaussian", "value": f"{config.PREPROCESS['gaussian_k']} px"},
                    ],
                },
                {
                    "title": "1.3 Phân ngưỡng thô",
                    "description": "Tách sơ bộ tiền cảnh/hậu cảnh bằng khoảng cách màu và Otsu (mặt nạ thô còn nhiễu hạt và lỗ thủng).",
                    "img": _encode_image(raw_mask_vis),
                    "metrics": [
                        {"label": "Thuật toán", "value": config.SEGMENT["method"]},
                        {"label": "Mô hình nền", "value": config.SEGMENT["bg_model"]},
                    ],
                },
                {
                    "title": "1.4 Làm sạch Morphology & Tách quả",
                    "description": "Phép Mở (Opening) xóa nhiễu ngoài nền, phép Đóng (Closing) vá lỗ trên quả và Connected Component giữ lại quả chính.",
                    "img": _encode_image(segmented_overlay),
                    "metrics": [
                        {"label": "Open / Close", "value": f"{config.SEGMENT['open_k']} / {config.SEGMENT['close_k']} px"},
                        {"label": "Diện tích", "value": f"{cv2.countNonZero(obj.mask):,} px"},
                    ],
                },
            ]
        },
        {
            "number": "02",
            "title": "Trích đặc trưng",
            "subtitle": "Màu chín · Khuyết tật · Hình dạng",
            "substep": [
                {
                    "title": "2.1 Đặc trưng màu sắc đạt chuẩn",
                    "description": "Tách vùng vỏ quả có sắc độ (Hue) và độ bão hòa đạt chuẩn chín bên trong mặt nạ quả.",
                    "img": _encode_image(ripe_overlay),
                    "metrics": [
                        {"label": "Màu đạt", "value": f"{features['ripe_ratio'] * 100:.1f}%"},
                        {"label": "Vùng xét", "value": "Bên trong Mask"},
                    ],
                },
                {
                    "title": "2.2 Phát hiện khuyết tật (Defect)",
                    "description": "Khoanh vùng các đốm thâm, dập nát hoặc mốc dựa trên độ lệch màu và cường độ sáng bất thường.",
                    "img": _encode_image(defect_overlay),
                    "metrics": [
                        {"label": "Vết hỏng", "value": f"{features['defect_ratio']:.1f}%"},
                        {"label": "Số điểm vết", "value": f"{features['n_spots']} vết"},
                    ],
                },
                {
                    "title": "2.3 Đặc trưng hình dạng (Shape)",
                    "description": "Trích xuất đường bao (Contour), khung bao (Bounding Box) và trọng tâm để tính độ tròn và diện tích.",
                    "img": _encode_image(shape_vis),
                    "metrics": [
                        {"label": "Circularity", "value": f"{features['circularity']:.3f}"},
                        {"label": "Diện tích", "value": f"{features['area_px']:,} px"},
                    ],
                },
            ]
        },
        {
            "number": "03",
            "title": "Phân loại",
            "subtitle": "Module 3",
            "description": "Luật Rule-based so sánh các ngưỡng đã hiệu chỉnh để chọn Loại 1, 2 hoặc 3.",
            "before": _encode_image(defect_overlay),
            "after": _encode_image(result),
            "metrics": [
                {"label": "Kết quả", "value": label},
                {"label": "Điểm vết", "value": f"{features['n_spots']} vết"},
                {"label": "Lý do", "value": reason},
            ],
        },
    ]

    return {
        "ok": True,
        "label": label,
        "fruit": fruit,
        "reason": reason,
        "features": {
            "ripe_ratio": round(float(features["ripe_ratio"]), 4),
            "defect_ratio": round(float(features["defect_ratio"]), 3),
            "n_spots": int(features["n_spots"]),
            "circularity": round(float(features["circularity"]), 4),
            "area_px": int(features["area_px"]),
        },
        "steps": steps,
    }
