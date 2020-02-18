r"""
Get root path & root config path & 
"""
import os.path as osp
import sys  # isort:skip

module_name = "debug"
p = __file__
while osp.basename(p) != module_name:
    p = osp.dirname(p)

ROOT_PATH = osp.dirname(p)
ROOT_CFG = osp.join(ROOT_PATH, 'config.yaml')
# sys.path.insert(0, ROOT_PATH)  # isort:skip
SRC_PATH = osp.join(ROOT_PATH, "src")
LIB_PATH = osp.join(ROOT_PATH, "src/lib")
DCN_PATH = osp.join(ROOT_PATH, "src/lib/models/networks/DCNv2")

sys.path.insert(0, SRC_PATH)  # isort:skip
sys.path.insert(0, LIB_PATH)  # isort:skip
sys.path.insert(0, DCN_PATH)  # isort:skip
