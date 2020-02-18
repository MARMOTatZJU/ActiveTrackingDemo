from .bbox import calc_IoU

class IOUTracker(object):
    def __init__(self, iou_thres=0., score_thres=0.):
        self.Ta = []
        # self.Tf = []
        self.id_cnt = 0
        self.iou_thres = iou_thres
        self.score_thres = score_thres
    
    def __call__(self, det_bboxes, ):
        det_idxs = list(range(len(det_bboxes)))
        target_ids = [-1 for _ in range(len(det_bboxes))]

        Ta_remove_idxs = []

        for Ta_idx in range(len(self.Ta)):
            if len(det_idxs) == 0:
                Ta_remove_idxs.append(Ta_idx)
                continue
            ious = [calc_IoU(self.Ta[Ta_idx]["bboxes"][-1][:4], det_bboxes[det_idx][:4]) for det_idx in det_idxs]
            best_idx = max(range(len(ious)), key=lambda idx: ious[idx])
            best_det_idx = det_idxs[best_idx]
            best_iou = ious[best_idx]
            best_score = det_bboxes[best_det_idx][4]

            if best_iou >= self.iou_thres:
                # match
                self.Ta[Ta_idx]["bboxes"].append(det_bboxes[best_det_idx])
                target_ids[best_det_idx] = self.Ta[Ta_idx]["track_id"]
                # from IPython import embed;embed()
                det_idxs.pop(best_idx)
            else:
                # dead
                Ta_remove_idxs.append(Ta_idx)
                # if best_score >= self.score_thres: 
        
        # remove disappearance
        sorted(Ta_remove_idxs)
        Ta_remove_idxs.reverse()
        for idx in Ta_remove_idxs:
            self.Ta.pop(idx)

        for det_idx in det_idxs:
            self.Ta.append(dict(track_id=self.id_cnt, 
                                bboxes=[det_bboxes[det_idx]]))
            target_ids[det_idx] = self.id_cnt
            self.id_cnt += 1

        return target_ids
