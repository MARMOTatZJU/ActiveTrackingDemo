from copy import deepcopy

import cv2

from _utils.video_reader import VideoReader

# vid_path = "/data/Videos/ActiveTrackingDemo/frisbee/frisbee_Trim03_2"
vid_path = "/data/Videos/ActiveTrackingDemo/football/football_Trim01"
video_reader = VideoReader(vid_path)

im = video_reader[0]
# bbox = [1000, 1000, 100, 100]

# im_ = deepcopy(im); cv2.rectangle(im_, (10, 10), (100, 100), (0, 255, 255),2);im_ = cv2.resize(im_, (0,0), fx=0.5, fy=0.5);cv2.imshow("preview", im_);cv2.waitKey(0)

from IPython import embed;embed()
