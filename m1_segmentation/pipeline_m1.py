"""MODULE 1 - tiền xử lý ảnh và tách các quả khỏi nền."""
import config
from common.contracts import FruitObject
from .preprocess import preprocess_frame
from .segment import segment_fruits
from .objects import extract_objects


def detect_fruits(frame_bgr, pre_cfg=None, seg_cfg=None):
    """
    Input : ảnh BGR uint8 gốc
    Output: (objects: List[FruitObject], mask_full: uint8 {0,255})
    """
    seg_cfg = seg_cfg or config.SEGMENT
    pre = preprocess_frame(frame_bgr, pre_cfg)
    mask = segment_fruits(pre, seg_cfg)
    objs = extract_objects(frame_bgr, mask, min_area=seg_cfg["min_area"])
    return objs, mask
