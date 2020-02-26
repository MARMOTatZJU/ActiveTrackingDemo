from typing import List, Tuple
from collections import OrderedDict

import numpy as np

from .bbox import calc_IoU, calc_centerness, is_pt_inside_bbox, xyxy2cxywh, cxywh2xyxy

_RELATIONSHIP_CLASSES = ["possessing by hand", "possessing by foot", 
# "hit by head"
]

class FastRelationshipDetector(object):
    _MAX_CNT = 5
    _TARGET_BBOX_EXPANSION_RATIO = 1.6

    def __init__(self,):
        self.relationship_histories = OrderedDict()

    def __call__(self, target_bbox: np.array, human_bboxes: np.array, human_ids: List[int]) -> Tuple[int, str]:
        r"""Detector
        
        Parameters
        ----------
        target_bbox : np.array
            target_bbox, 
            able to be reshape to (4,)
        human_bbox : np.array
            N detected humans with their
            able to be reshape to (N, 39)
            2nd channel's order: x1, y1, x2, y2, confidence, (x, y)*17 keypoint (coco definition)
        human_ids : List[int]
            List of human ids.
            Share indexes with human_bboxes 

        Retunrns
        --------
        Tuple[int, str]
            index & relation type
        """
    
        target_bbox = self._expand_bbox(target_bbox)

        human_bboxes = np.array(human_bboxes).reshape(-1, 39)
        L = len(human_bboxes)  # detection number
        if L <= 0:
            human_id = -1
            relation_type = "no relationship"
            return human_id, relation_type

        human_keypoints = human_bboxes[:, 5:].reshape(L, 17, 2)

        # (L, 2)
        left_hand_pt = human_keypoints[:, 9, :]
        right_hand_pt = human_keypoints[:, 10, :]
        left_foot_pt = human_keypoints[:, 15, :]
        right_foot_pt = human_keypoints[:, 16, :]

        # (L,)
        left_hand_ctr = calc_centerness(left_hand_pt, target_bbox)
        right_hand_ctr = calc_centerness(right_hand_pt, target_bbox)
        left_foot_ctr = calc_centerness(left_foot_pt, target_bbox)
        right_foot_ctr = calc_centerness(right_foot_pt, target_bbox)

        # (L, 4)
        ctrs = np.stack([left_hand_ctr, right_hand_ctr, left_foot_ctr, right_foot_ctr], axis=1)
        # (L)
        max_ctrs = ctrs.max(axis=1)
        argmax_ctr_idx = max(range(len(max_ctrs)), 
                             key=lambda idx: max_ctrs[idx])

        if max_ctrs[argmax_ctr_idx] > 0:
            human_id = human_ids[argmax_ctr_idx]
            max_hand_ctr = max(left_hand_ctr[argmax_ctr_idx], right_hand_ctr[argmax_ctr_idx])
            max_foot_ctr = max(left_foot_ctr[argmax_ctr_idx], right_foot_ctr[argmax_ctr_idx])
            if max_hand_ctr > max_foot_ctr:
                relation_type = "possessing by hand"
            else:
                relation_type = "possessing by foot"
        else:
            human_id = -1
            relation_type = "no relationship"

        return human_id, relation_type
    
    def _expand_bbox(self, target_bbox):
        target_bbox = np.array(target_bbox).reshape(4)
        target_bbox = xyxy2cxywh(target_bbox)
        target_bbox[2:] *= self._TARGET_BBOX_EXPANSION_RATIO
        target_bbox = cxywh2xyxy(target_bbox)
        return target_bbox
