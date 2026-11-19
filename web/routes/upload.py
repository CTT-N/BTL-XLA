from m3_classify_viz.pipeline import run_pipeline
from m3_classify_viz.make_video import make_conveyor_video, select_from_manifest

from urllib.parse import quote
import cv2
import config
import numpy as np
import time
from pathlib import Path
from flask import Blueprint, jsonify, request, send_file


# Import các đường dẫn và hàm xử lý từ file gốc hoặc module services
from web.services import DATA_DIR, analyze_image, UPLOAD_DIR

# Tạo Blueprint
upload_bp = Blueprint("upload", __name__)

@upload_bp.route("/api/analyze-upload", methods=["POST"])
def analyze_upload():
    fruit = request.form.get("fruit", "apple")
    if fruit not in config.FRUIT_CFG:
        return jsonify({"ok": False, "error": "Loại trái cây không hợp lệ."}), 400
    file = request.files.get("image")
    if file is None or not file.filename:
        return jsonify({"ok": False, "error": "Chưa chọn tệp ảnh."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        return jsonify({"ok": False, "error": "Chỉ hỗ trợ JPG, PNG, BMP hoặc WEBP."}), 400

    data = file.read()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"ok": False, "error": "Không thể giải mã ảnh."}), 400
    return jsonify(analyze_image(image, fruit))

@upload_bp.route("/uploads/<path:filename>")
def uploaded_file(filename):
    path = UPLOAD_DIR / filename
    if not path.exists():
        return "Không tìm thấy tệp", 404
    mime = "video/webm" if path.suffix.lower() == ".webm" else "video/mp4"
    return send_file(path, mimetype=mime)

@upload_bp.route("/api/demo-video", methods=["POST"])
def demo_video_route():
    """Tạo nhanh video băng chuyền mô phỏng 6 quả từ dataset test và chạy pipeline M1->M2->M3."""
    data = request.get_json(silent=True) or {}
    fruit = data.get("fruit", "apple")
    if fruit not in config.FRUIT_CFG:
        return jsonify({"ok": False, "error": "Loại trái cây không hợp lệ."}), 400

    ts = int(time.time())
    sim_path = UPLOAD_DIR / f"sim_{fruit}_{ts}.mp4"
    out_name = f"out_sim_{fruit}_{ts}.webm"
    out_path = UPLOAD_DIR / out_name

    items = select_from_manifest(
        str(DATA_DIR / "manifest.csv"), str(DATA_DIR), fruit, "test", n_per_label=3, seed=ts % 100
    )
    if not items:
        return jsonify({"ok": False, "error": "Không đủ ảnh trong dataset để tạo video."}), 400

    make_conveyor_video(items, str(sim_path))
    res = run_pipeline(str(sim_path), str(out_path), fruit=fruit, verbose=False)
    return jsonify({
        "ok": True,
        "video_url": f"/uploads/{quote(out_name)}",
        "fps": round(float(res["fps"]), 1),
        "n_frames": int(res["n_frames"]),
        "results": res["results"],
    })
