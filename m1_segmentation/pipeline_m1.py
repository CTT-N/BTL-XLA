"""
MODULE 1 - HÀM GỘP MÀ M3 GỌI:  detect_fruits(frame)  rồi  tracker.update(objs)
"""
from typing import List, Tuple, overload, Literal
import numpy as np
import config
from common.contracts import FruitObject
from .preprocess import preprocess_frame
from .segment import segment_fruits
from .objects import extract_objects


@overload
def detect_fruits(
    frame_bgr: np.ndarray, frame_idx: int = 0, pre_cfg=None, seg_cfg=None, *, return_debug: Literal[True]
) -> Tuple[List[FruitObject], np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def detect_fruits(
    frame_bgr: np.ndarray, frame_idx: int = 0, pre_cfg=None, seg_cfg=None, return_debug: Literal[False] = False
) -> Tuple[List[FruitObject], np.ndarray]: ...


def detect_fruits(frame_bgr, frame_idx=0, pre_cfg=None, seg_cfg=None, return_debug=False):
    seg_cfg = seg_cfg or config.SEGMENT
    pre = preprocess_frame(frame_bgr, pre_cfg)
    if return_debug:
        mask, raw_mask = segment_fruits(pre, seg_cfg, return_raw=True)
        objs = extract_objects(frame_bgr, mask, frame_idx, seg_cfg["min_area"])
        return objs, mask, pre, raw_mask

    mask = segment_fruits(pre, seg_cfg)
    objs = extract_objects(frame_bgr, mask, frame_idx, seg_cfg["min_area"])
    return objs, mask