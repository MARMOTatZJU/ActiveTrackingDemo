# import importlib
# from utils.misc import import_module
from tqdm import tqdm

import cv2
from _utils.video_writer import VideoWriter
from _utils.iou_tracker import IOUTracker
from _utils.bbox import calc_IoU

# dataset = import_module("dataset", "/home/lan/Documents/xuyinda/Projects/video_analyst/debug/import_tracker.py")
# dataset = import_module("dataset", "/home/lan/Documents/xuyinda/Projects/video_analyst/debug/import_dataset.py")
# tracker = importlib.util.spec_from_file_location("tracker", "/home/lan/Documents/xuyinda/Projects/video_analyst/debug/import_tracker.py")
# pose_detector = importlib.util.spec_from_file_location("pose_detector", "/home/lan/Documents/xuyinda/Projects/CenterNet/debug/import_pose_detector.py")

# video_name = "frisbee"
# video_name = "handball1"
# video_name = "handball2"
# video_name = "ball1"
# video_names = ["handball1", "ball1", "basketball", "tiger"]

import sys
# sys.path.append("/home/lan/Documents/xuyinda/Projects/video_analyst/debug")
# from import_dataset import dataset, vot_benchmark
# from import_tracker import tracker, xywh2xyxy
from import_videoanalyst import dataset, vot_benchmark
from import_videoanalyst import tracker, xywh2xyxy

# sys.path.append("/home/lan/Documents/xuyinda/Projects/CenterNet/debug")
# from import_pose_detector import pose_detector, Debugger
from import_centernet import pose_detector, Debugger

imresize_ratio = 1
# sys.path.append("/home/lan/Documents/xuyinda/Projects/CenterNet/src/lib/models/networks/DCNv2")

# video_names = ["frisbee", "book", "ball1"]
# video_names = ["glove"]
video_names = ["frisbee"]
for video_name in video_names:
    print("Run %s"%video_name)
    video_file = "./%s.avi"%video_name
    video_writer = VideoWriter(video_file, fps=20, scale=0.6)

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
    person_tracker = IOUTracker()
    
    elapsed_time = 0
    for idx in tqdm(range(1, len_video)):
        # fetch frame image
        image_file = image_files[idx]
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
        person_target_ious = [calc_IoU(target_bbox, bbox[:4]) for bbox in results[1]]
        if (len(person_target_ious)==0) or (max(person_target_ious) <= 0):
            holder_id = -1
        else:
            holder_id = person_ids[max(range(len(person_ids)), key=lambda idx: person_target_ious[idx])]
        # draw object target_bbox
        target_bbox = tuple(map(int, target_bbox))
        color = (0, 255, 255) if (holder_id != -1) else (0, 127, 255)
        cv2.rectangle(debugger.imgs["multi_pose"], target_bbox[:2], target_bbox[2:], color)

        # draw interaction status
        txt = "holder_id %d"%holder_id
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
        cv2.rectangle(debugger.imgs["multi_pose"],
            (target_bbox[0], target_bbox[1] - cat_size[1] - 2),
            (target_bbox[0] + cat_size[0], target_bbox[1] - 2), color, -1)
        cv2.putText(debugger.imgs["multi_pose"], txt, (target_bbox[0], target_bbox[1] - 2), 
                    font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        # visualize
        im_preview = debugger.imgs["multi_pose"]
        video_writer.write(im_preview)
        # cv2.imshow("preview", im_preview)
        # cv2.waitKey(0)
        # from IPython import embed;embed()
    FPS = (len_video - 1) / elapsed_time
    print("FPS: %.2f"%FPS)