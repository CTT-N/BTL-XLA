"""
Chuẩn bị cho module 3: extract_all_features(obj, fruit)
"""
from typing import Any, Dict, Literal, Tuple, overload
import cv2
import numpy as np
import config
from common.contracts import FruitObject, FEATURE_KEYS, validate_features
from .color_features import extract_color_features, ripe_pixel_mask
from .defect_detection import detect_defects
from .shape_features import extract_shape_features


@overload
def extract_all_features(
    obj: FruitObject, fruit: str = "apple", cfg=None, *, return_debug: Literal[True]
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]: ...


@overload
def extract_all_features(
    obj: FruitObject, fruit: str = "apple", cfg=None, return_debug: Literal[False] = False
) -> Dict[str, Any]: ...

def extract_all_features(obj: FruitObject, fruit: str = "apple", cfg=None, return_debug: bool = False):
    """
    Input : FruitObject (từ M1), fruit = 'apple' | 'orange'
    Output: feature dict (hoặc (feat, debug_masks) nếu return_debug=True)
    """
    fruit_cfg = config.FRUIT_CFG[fruit]
    color = extract_color_features(obj.crop_bgr, obj.mask, fruit, fruit_cfg)
    defect = detect_defects(obj.crop_bgr, obj.mask, cfg or config.DEFECT, fruit_cfg)
    shape = extract_shape_features(obj.mask)

    raw = {**color, **defect, **shape}
    feat = {k: raw[k] for k in FEATURE_KEYS}
    validate_features(feat)

    if not return_debug:
        return feat

    hsv = cv2.cvtColor(obj.crop_bgr, cv2.COLOR_BGR2HSV)
    ripe_mask = (ripe_pixel_mask(hsv[..., 0], hsv[..., 1], hsv[..., 2], fruit_cfg).astype(np.uint8) * 255) & obj.mask
    return feat, {"ripe_mask": ripe_mask, "defect_mask": feat["defect_mask"]}