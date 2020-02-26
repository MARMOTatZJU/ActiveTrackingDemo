# -*- coding: utf-8 -*
import os
import os.path as osp
import json

import glob

import numpy as np
import cv2

from .bbox import xyxy2xywh

class VideoReader(object):
    
    def __init__(self, video_dir):
        self.video_dir = video_dir
        self.img_paths = glob.glob(osp.join(self.video_dir, "*.jpg"))
        self.img_paths = sorted(self.img_paths)

        anno_path = osp.join(self.video_dir, "annotation.json")
        with open(anno_path, "r") as f:
            self.anno = json.load(f)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        im = cv2.imread(img_path)
        return im
    
    def __len__(self):
        return len(self.img_paths)
