"""Kiểm thử luật phân loại và luồng xử lý từng ảnh tĩnh."""
from common.contracts import GRADE_1, GRADE_2, GRADE_3, GRADES
from m1_segmentation.pipeline_m1 import detect_fruits
from m1_segmentation.test_m1 import build_still
from m2_features.extract_all import extract_all_features
from .classifier import classify_rule_based


def test_classifier():
    ok = dict(circularity=0.95, ripe_ratio=0.95)
    assert classify_rule_based(dict(defect_ratio=0.2, **ok))[0] == GRADE_1
    assert classify_rule_based(dict(defect_ratio=4.0, **ok))[0] == GRADE_2
    assert classify_rule_based(dict(defect_ratio=25.0, **ok))[0] == GRADE_3
    assert classify_rule_based(dict(defect_ratio=0.2, circularity=0.6, ripe_ratio=0.9))[0] == GRADE_3
    assert classify_rule_based(dict(defect_ratio=0.2, circularity=0.8, ripe_ratio=0.9))[0] == GRADE_2
    print("[OK] luật phân loại")


def test_static_image_pipeline():
    cases = [
        ("apple", "apple_red", {}),
        ("apple", "apple_red", {"n_spots": 2, "spot_radius": 22}),
        ("orange", "orange", {"rot_fraction": 0.3}),
    ]
    for i, (fruit, kind, options) in enumerate(cases):
        image, _ = build_still(160, kind, vg=0.1, seed=i, **options)
        objects, _ = detect_fruits(image)
        assert objects, f"M1 did not segment the {fruit} image"

        features = extract_all_features(max(objects, key=lambda obj: obj.mask.sum()), fruit)
        label, reason = classify_rule_based(features)
        assert label in GRADES
        assert isinstance(reason, str) and reason
        print(f"[OK] {fruit} static image -> {label}")


if __name__ == "__main__":
    test_classifier()
    test_static_image_pipeline()
    print("\n>>> TẤT CẢ KIỂM THỬ ẢNH TĨNH ĐẠT")
