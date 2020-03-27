# import importlib
# from utils.misc import import_module
from tqdm import tqdm

import cv2
from _utils.video_writer import VideoWriter
from _utils.iou_tracker import IOUTracker
# from _utils.relationship_detector import RelationshipDetector
from _utils.fast_relationship_detector import FastRelationshipDetector
from _utils.bbox import calc_IoU


# video_name = "frisbee"
# video_name = "handball1"
# video_name = "handball2"
# video_name = "ball1"
# video_names = ["handball1", "ball1", "basketball", "tiger"]

import sys
from import_videoanalyst import dataset, vot_benchmark
from import_videoanalyst import tracker, xywh2xyxy

from import_centernet import pose_detector, Debugger

imresize_ratio = 1
video_writer_scale = 0.4

# video_names = ["frisbee", "book", "ball1"]
# video_names = ["glove"]
video_names = ["frisbee"]
for video_name in video_names:
    print("Run %s"%video_name)
    video_file = "./%s.avi"%video_name
    video_writer = VideoWriter(video_file, fps=20, scale=video_writer_scale)

    video = dataset[video_name]
    image_files, gt = video['image_files'], video['gt']
    len_video = len(image_files)

    # inittialize tracker
    image_file = image_files[0]
    im = vot_benchmark.get_img(image_file)
    im = cv2.resize(im, (0, 0), fx=imresize_ratio, fy=imresize_ratio)
    cx, cy, w, h = vot_benchmark.get_axis_aligned_bbox(gt[0])
    rect = vot_benchmark.cxy_wh_2_rect((cx, cy), (w, h))
    rect = rect*imresize_ratio

    tracker.init(im, rect)
    # dump template
    im_template = tracker._state["z_crop"]
    p_dump = "./tmp/template.jpg"
    cv2.imwrite(p_dump, im_template)
    print(p_dump)

    person_tracker = IOUTracker()
    rel_detector = FastRelationshipDetector()
    
    elapsed_time = 0
    for frame_idx in tqdm(range(1, len_video)):
        # fetch frame image
        image_file = image_files[frame_idx]
        im = vot_benchmark.get_img(image_file)
        im = cv2.resize(im, (0, 0), fx=imresize_ratio, fy=imresize_ratio)
        # run pose detector
        tick_start = cv2.getTickCount()
        ret = pose_detector.run(im)
        elapsed_time += (cv2.getTickCount() - tick_start) / cv2.getTickFrequency()
        results = ret["results"]
        # filter bbox with low confidence
        results[1] = [bbox for bbox in results[1] if bbox[4] > pose_detector.opt.vis_thresh]
        # IoU tracker track person id
        person_ids = person_tracker([bbox[:5] for bbox in results[1]])
        # from IPython import embed;embed()

        # draw keypoints
        debugger = Debugger(dataset=pose_detector.opt.dataset, ipynb=(pose_detector.opt.debug==3),
                            theme=pose_detector.opt.debugger_theme)
        debugger.add_img(im, img_id='multi_pose')
        for idx, bbox in enumerate(results[1]):
            # if bbox[4] > pose_detector.opt.vis_thresh:
            prompt = "person_id %d"%person_ids[idx]
            debugger.add_coco_bbox(bbox[:4], 0, bbox[4], img_id='multi_pose', prompt=prompt)
            debugger.add_coco_hp(bbox[5:39], img_id='multi_pose')


        # run tracker
        tick_start = cv2.getTickCount()
        rect = tracker.update(im)
        elapsed_time += (cv2.getTickCount() - tick_start) / cv2.getTickFrequency()
        target_bbox = xywh2xyxy(rect)

 
        # inference for holder
        holder_id, relation_type = rel_detector(target_bbox, results[1], person_ids)
        # from IPython import embed;embed()
        # print(holder_id)
 
        # inference for holder
        # person_target_ious = [calc_IoU(target_bbox, bbox[:4]) for bbox in results[1]]
        # if (len(person_target_ious)==0) or (max(person_target_ious) <= 0):
        #     holder_id = -1
        # else:
        #     # argmax_{person_target_ious} person_ids
        #     holder_id = person_ids[max(range(len(person_ids)), key=lambda idx: person_target_ious[idx])]


        # draw object target_bbox
        target_bbox = tuple(map(int, target_bbox))
        color = (0, 255, 255) if (holder_id != -1) else (0, 127, 255)
        cv2.rectangle(debugger.imgs["multi_pose"], target_bbox[:2], target_bbox[2:], color)

        # draw interaction status
        txt = "holder_id %03d"%holder_id
        txt1 = relation_type
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
        cat_size1 = cv2.getTextSize(txt1, font, 0.5, 2)[0]
        # draw for txt
        cv2.rectangle(debugger.imgs["multi_pose"],
            (target_bbox[0], target_bbox[1] - cat_size[1] - cat_size1[1] - 2 - 2),
            (target_bbox[0] + cat_size[0], target_bbox[1] - cat_size1[1] - 2 - 2), color, -1)
        cv2.putText(debugger.imgs["multi_pose"], txt, (target_bbox[0], target_bbox[1] - cat_size1[1] - 2 - 2), 
                    font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        # draw for txt1
        cv2.rectangle(debugger.imgs["multi_pose"],
            (target_bbox[0], target_bbox[1] - cat_size1[1] - 2),
            (target_bbox[0] + cat_size1[0], target_bbox[1] - 2), color, -1)
        cv2.putText(debugger.imgs["multi_pose"], txt1, (target_bbox[0], target_bbox[1] - 2), 
                    font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        # visualize
        im_preview = debugger.imgs["multi_pose"]
        video_writer.write(im_preview)

        # DEBUG
        dump_freq = 1
        if frame_idx % dump_freq == 0:
            p_dump = "./tmp/%05d.jpg"%frame_idx
            cv2.imwrite(p_dump, im_preview)
            print(p_dump)
        # DEBUG

        # cv2.imshow("preview", im_preview)
        # cv2.waitKey(0)
        # from IPython import embed;embed()
    FPS = (len_video - 1) / elapsed_time
    print("FPS: %.2f"%FPS)
