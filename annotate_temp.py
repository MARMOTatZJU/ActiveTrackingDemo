from copy import deepcopy

import cv2

from _utils.video_reader import VideoReader

vid_path = "/data/Videos/ActiveTrackingDemo/frisbee/frisbee_Trim01"
video_reader = VideoReader(vid_path)

im = video_reader[0]
# bbox = [1000, 1000, 100, 100]

# im_ = deepcopy(im); cv2.rectangle(im_, (545, 499), (638, 533), (0, 255, 255),2); cv2.imshow("preview", im_);cv2.waitKey(0)
# im_ = deepcopy(im); cv2.rectangle(im_, (1000, 500), (1100, 600), (0, 255, 255),2); cv2.imshow("preview", im_);cv2.waitKey(0)

from IPython import embed;embed()
