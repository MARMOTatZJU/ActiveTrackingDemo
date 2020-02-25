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

import os.path as osp
import logging

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task

from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.pipeline.utils.bbox import xywh2xyxy
from videoanalyst.utils import complete_path_wt_root_in_cfg

import torch

logger = logging.getLogger('global')

# exp_cfg_path = osp.realpath("/home/lan/Documents/xuyinda/Projects/video_analyst/experiments/siamfcpp/test/siamfcpp_googlenet.yaml")
exp_cfg_path = osp.join(ROOT_PATH, "experiments/siamfcpp/test/vot/siamfcpp_googlenet-new.yaml")
# exp_cfg_path = osp.join(ROOT_PATH, "experiments/siamfcpp/test/vot/siamfcpp_googlenet-multi_temp.yaml")
# exp_cfg_path = osp.join(ROOT_PATH, "experiments/siamfcpp/test/vot/siamfcpp_alexnet-new.yaml")
# exp_cfg_path = osp.join(ROOT_PATH, "experiments/siamfcpp/test/vot/siamfcpp_alexnet.yaml")
# exp_cfg_path = osp.join(ROOT_PATH, "experiments/siamfcpp/test/vot/siamfcpp_tinyconv.yaml")
root_cfg.merge_from_file(exp_cfg_path)
logger.info("Load experiment config. at: %s" % exp_cfg_path)

# resolve config
root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
root_cfg = root_cfg.test
task, task_cfg = specify_task(root_cfg)
task_cfg.freeze()

# build model
model = model_builder.build(task, task_cfg.model)
# build pipeline
# pipeline = pipeline_builder.build_pipeline('track', task_cfg.pipeline)
# pipeline.set_model(model)
pipeline = pipeline_builder.build_pipeline(task, task_cfg.pipeline, model)
dev = torch.device("cuda:0")
pipeline.to_device(dev)

tracker = pipeline

if __name__ == "__main__":
    from IPython import embed;embed()
