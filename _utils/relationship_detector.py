from typing import List, Tuple
from collections import OrderedDict

import numpy as np

from .bbox import calc_IoU, is_pt_inside_bbox, xyxy2cxywh, cxywh2xyxy

_RELATIONSHIP_CLASSES = ["possessing by hand", "possessing by foot", 
# "hit by head"
]

class RelationshipDetector(object):
    _MAX_CNT = 5
    _TARGET_BBOX_EXPANSION_RATIO = 1.2

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
        target_bbox = np.array(target_bbox).reshape(4)
        target_bbox = xyxy2cxywh(target_bbox)
        target_bbox[2:] *= self._TARGET_BBOX_EXPANSION_RATIO
        target_bbox = cxywh2xyxy(target_bbox)
        

        human_bboxes = np.array(human_bboxes).reshape(-1, 39)
        L = len(human_bboxes)  # detection number
        # idxs = list(range(L))

        human_keypoints = human_bboxes[:, 5:].reshape(L, 17, 2)

        person_target_ious = [calc_IoU(target_bbox, bbox[:4]) for bbox in human_bboxes]
        # recall those
        pos_iou_idxs = [i for i in range(L) if person_target_ious[i] > 0]
        
# no relationship, return
        if len(pos_iou_idxs) <= 0:
            # print("no iou")
            self._decay_hist()
            return -1, ""

        has_relation_idxs = []
        relation_types = []
        for idx in pos_iou_idxs:
            keypoints = human_keypoints[idx]
            left_hand_pt = keypoints[9]
            right_hand_pt = keypoints[10]
            left_foot_pt = keypoints[15]
            right_foot_pt = keypoints[16]
            # head_pts = [keypoints[0], keypoints[1], keypoints[2], keypoints[3], keypoints[4]]
            # inference rule: keypoint in bbox
            possess_by_hand = is_pt_inside_bbox(left_hand_pt, target_bbox) or is_pt_inside_bbox(right_hand_pt, target_bbox)
            possess_by_foot = is_pt_inside_bbox(left_foot_pt, target_bbox) or is_pt_inside_bbox(right_foot_pt, target_bbox)

            if possess_by_hand:
                relation_type = "possessing by hand"
            elif possess_by_foot:
                relation_type = "possessing by foot"
            else:
                relation_type = None
            # from IPython import embed;embed()
            if relation_type is not None:
                has_relation_idxs.append(idx)
                relation_types.append(relation_type)

# no relationship, return
        if len(has_relation_idxs) <= 0:
            self._decay_hist()
            # print("no relation")
            return -1, ""
# unique instance with relationship, return it
        elif len(has_relation_idxs) == 1:
            human_id = human_ids[has_relation_idxs[0]]
            self._add_count_hist(human_id)
            # print("unique relation")

            return human_id, relation_types[0]

# multiple instances with relationship, check history
        hist_cnts = [self.relationship_histories[human_ids[idx]] for idx in has_relation_idxs]
        hist_cnt_max = max(hist_cnts)

        # has_raltion_idxs & relation_types
        all_idxs = list(range(len(has_relation_idxs)))

        cand_idxs = [idx for idx in all_idxs if self.relationship_histories[human_ids[all_idxs[idx]]] == hist_cnt_max]
    # non-deterministic choice
        rand_idx = np.random.choice(cand_idxs)
        idx = all_idxs[rand_idx]

        human_id = human_ids[idx]
        self._add_count_hist(human_ids[idx])

        return human_id, relation_types[idx]



    def _decay_hist(self):
        keys = self.relationship_histories.keys()
        for k in keys:
            self.relationship_histories[k] -= 1
            # histroy dies
            if self.relationship_histories[k] <= 0:
                del self.relationship_histories[k]

    def _add_count_hist(self, k):
        if k not in self.relationship_histories:
            self.relationship_histories[k] = 1
        else:
            self.relationship_histories[k] += 1

        self.relationship_histories[k] = min(self._MAX_CNT, self.relationship_histories[k])



