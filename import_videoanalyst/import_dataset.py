# from .paths import ROOT_PATH  # isort:skip

import os.path as osp
import sys  # isort:skip

# module_name = "debug"
# p = __file__
# while osp.basename(p) != module_name:
#     p = osp.dirname(p)
p = osp.dirname(osp.realpath(__file__))
p = osp.dirname(p)
p = osp.join(p, "third_party", "video_analyst")

ROOT_PATH = p
# ROOT_CFG = osp.join(ROOT_PATH, 'config.yaml')
sys.path.insert(0, ROOT_PATH)  # isort:skip

# ============== path ============== #

import cv2

from videoanalyst.evaluation import vot_benchmark

DATASET_DIR = osp.join(ROOT_PATH, "datasets/VOT/vot2018")
# dataset = vot_benchmark.load_dataset("/home/lan/Documents/xuyinda/Projects/video_analyst/datasets/VOT/vot2018", "VOT2018")
dataset = vot_benchmark.load_dataset(DATASET_DIR, "VOT2018")

if __name__ == "__main__":
    from IPython import embed;embed()
