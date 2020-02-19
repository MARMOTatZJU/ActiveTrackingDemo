# from paths import LIB_PATH

import os.path as osp
import sys  # isort:skip

# module_name = "debug"
# p = __file__
# while osp.basename(p) != module_name:
#     p = osp.dirname(p)
p = osp.dirname(osp.realpath(__file__))
p = osp.dirname(p)
p = osp.join(p, "3rdparty", "CenterNet")

ROOT_PATH = p
# i = osp.join(ROOT_PATH, 'config.yaml')
# sys.path.insert(0, ROOT_PATH)  # isort:skip
SRC_PATH = osp.join(ROOT_PATH, "src")
LIB_PATH = osp.join(ROOT_PATH, "src/lib")
DCN_PATH = osp.join(ROOT_PATH, "src/lib/models/networks/DCNv2")

sys.path.insert(0, SRC_PATH)  # isort:skip
sys.path.insert(0, LIB_PATH)  # isort:skip
sys.path.insert(0, DCN_PATH)  # isort:skip

# ============== path ============== #

import os
import os.path as osp

from opts import opts
from detectors.detector_factory import detector_factory
from utils.debugger import Debugger

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

MODEL_DIR = osp.join(ROOT_PATH, "models/multi_pose_dla_3x.pth")

opt = opts().init(["multi_pose", 
                   "--load_model", MODEL_DIR, 
                   "--debug", "0"
                   ]) 

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
# opt.debug = max(opt.debug, 1)
Detector = detector_factory[opt.task]
pose_detector = Detector(opt)


if __name__ == "__main__":
    from IPython import embed;embed()
