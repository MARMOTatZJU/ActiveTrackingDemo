# from .paths import ROOT_PATH  # isort:skip

import os.path as osp
import sys  # isort:skip

module_name = "debug"
p = __file__
while osp.basename(p) != module_name:
    p = osp.dirname(p)

ROOT_PATH = osp.dirname(p)
ROOT_CFG = osp.join(ROOT_PATH, 'config.yaml')
sys.path.insert(0, ROOT_PATH)  # isort:skip

# ============== path ============== #

import os.path as osp
import logging

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task

from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.pipeline.utils.bbox import xywh2xyxy

logger = logging.getLogger('global')

exp_cfg_path = osp.realpath("/home/lan/Documents/xuyinda/Projects/video_analyst/experiments/siamfcpp/test/siamfcpp_googlenet.yaml")
root_cfg.merge_from_file(exp_cfg_path)
logger.info("Load experiment config. at: %s" % exp_cfg_path)

# resolve config
task, task_cfg = specify_task(root_cfg)
task_cfg.freeze()

# build model
model = model_builder.build_model(task, task_cfg.model)
# build pipeline
pipeline = pipeline_builder.build_pipeline('track', task_cfg.pipeline)
pipeline.set_model(model)
tracker = pipeline

if __name__ == "__main__":
    from IPython import embed;embed()
