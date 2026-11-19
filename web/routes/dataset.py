import mimetypes
from urllib.parse import quote
import cv2
from flask import Blueprint, jsonify, request, send_file

import config

# Import các đường dẫn và hàm xử lý từ file gốc hoặc module services
from web.services import DATA_DIR, ROOT, _dataset_images, analyze_image

# Tạo Blueprint
dataset_bp = Blueprint("dataset", __name__)


@dataset_bp.route("/api/dataset")
def dataset():
    fruit = request.args.get("fruit", "apple")
    if fruit not in config.FRUIT_CFG:
        fruit = "apple"
    images = _dataset_images(fruit)

    grouped = {"healthy": [], "rotten": []}
    for p in images:
        lbl = p.parent.name.lower()
        item = {
            "name": p.name,
            "label": lbl,
            "url": f"/dataset/{quote(fruit)}/{quote(p.parent.name)}/{quote(p.name)}",
            "path": str(p.relative_to(ROOT)).replace("\\", "/"),
        }
        if lbl in grouped:
            grouped[lbl].append(item)

    return jsonify({
        "fruit": fruit,
        "healthy": grouped["healthy"],
        "rotten": grouped["rotten"],
        "total": len(images),
    })


@dataset_bp.route("/dataset/<fruit>/<label>/<filename>")
def dataset_image(fruit, label, filename):
    path = DATA_DIR / "test" / fruit / label / filename
    if not path.exists():
        return "Không tìm thấy ảnh", 404
    mimetype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return send_file(path, mimetype=mimetype)


@dataset_bp.route("/api/analyze-dataset", methods=["POST"])
def analyze_dataset():
    data = request.get_json(silent=True) or {}
    fruit = data.get("fruit", "apple")
    path = data.get("path")
    if fruit not in config.FRUIT_CFG:
        return jsonify({"ok": False, "error": "Loại trái cây không hợp lệ."}), 400
    resolved = (ROOT / path).resolve() if path else None
    if resolved is None or DATA_DIR.resolve() not in resolved.parents or not resolved.exists():
        return jsonify({"ok": False, "error": "Đường dẫn ảnh không hợp lệ."}), 400
    image = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"ok": False, "error": "Không thể đọc ảnh."}), 400
    return jsonify(analyze_image(image, fruit))