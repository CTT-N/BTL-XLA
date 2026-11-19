from __future__ import absolute_import

import io
import mimetypes
import os
import random
from pathlib import Path
from urllib.parse import quote
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

import config
from m1_segmentation.pipeline_m1 import detect_fruits
from m2_features.extract_all import extract_all_features



WEB_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = WEB_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(WEB_DIR / "templates"),
    static_folder=str(WEB_DIR / "assets"),
)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


def _encode_image(image):
    # OpenCV stores image arrays in BGR, while browsers and PNG data are RGB.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ok, buf = cv2.imencode(".png", rgb_image)
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


def _object_for_image(image_bgr, fruit):
    objs, mask = detect_fruits(image_bgr)
    if not objs:
        return None, mask
    return max(objs, key=lambda obj: cv2.countNonZero(obj.mask)), mask

def _dataset_images(fruit):
    test_dir = DATA_DIR / "test" / fruit
    if not test_dir.exists():
        return []
    return sorted(test_dir.glob("*/*.png"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dataset")
def dataset():
    fruit = request.args.get("fruit", "apple")
    if fruit not in config.FRUIT_CFG:
        fruit = "apple"
    images = _dataset_images(fruit)
    sample = random.sample(images, min(12, len(images)))
    return jsonify({
        "fruit": fruit,
        "images": [
            {
                "name": p.name,
                "label": p.parent.name,
                "url": f"/dataset/{quote(fruit)}/{quote(p.parent.name)}/{quote(p.name)}",
                "path": str(p.relative_to(ROOT)).replace("\\", "/"),
            }
            for p in sample
        ],
    })


@app.route("/dataset/<fruit>/<label>/<filename>")
def dataset_image(fruit, label, filename):
    path = DATA_DIR / "test" / fruit / label / filename
    if not path.exists():
        return "Không tìm thấy ảnh", 404
    mimetype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    return send_file(path, mimetype=mimetype)



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
