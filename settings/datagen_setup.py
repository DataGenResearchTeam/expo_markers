import os
from detectron2.config import get_cfg
from detectron2 import model_zoo
from datetime import datetime
from detectron2.engine import default_setup
from detectron2.engine import default_argument_parser
from data.prepare_datasets import DATASET_DICTS
import json


def get_datagen_cfg():
    """add placeholders to enable later loading of custom attributes
       from config_source in [args.config_file, datagen_config.json, args.opts]"""
    cfg = get_cfg()
    cfg.DG_DATASET_SIZE = -1
    cfg.DG_SYNT_DATA_PATH = ""
    cfg.DG_REAL_DATA_PATH = ""
    cfg.DG_EVAL_OUTPUT_PATH = ""
    cfg.DG_TESTSET_TYPE = ""  # ['real', 'synt']
    cfg.DG_TESTSET_NAME = ""  # ['train', 'val', 'test']
    cfg.DG_DATA_TYPE = ""
    return cfg


def get_datagen_config():
    config_path = "settings/datagen_config.json"
    with open(config_path, "rb") as f:
        datagen_config = json.load(f)
    return datagen_config


def get_output_dir(cfg, datagen_config):
    if cfg.OUTPUT_DIR == "./output":  # the case where no outputdir was given (default value)
        if "output_dir" in datagen_config:
            return datagen_config["DG_OUTPUT_DIR"]
        else:
            current_time_str = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
            return "./output_{}".format(current_time_str)
    else:
        return cfg.OUTPUT_DIR


def update_from_datagen_config(cfg, args):
    datagen_config = get_datagen_config()
    config_file = model_zoo.get_config_file(datagen_config["d2_config_file"])
    cfg.merge_from_file(config_file)
    cfg.OUTPUT_DIR = get_output_dir(cfg, datagen_config)
    cfg.DG_EVAL_OUTPUT_PATH = datagen_config["DG_EVAL_OUTPUT_PATH"]
    cfg.DG_DATASET_SIZE = datagen_config["DG_DATASET_SIZE"]
    cfg.DG_SYNT_DATA_PATH = datagen_config["DG_SYNT_DATA_PATH"]
    cfg.DG_REAL_DATA_PATH = datagen_config["DG_REAL_DATA_PATH"]
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(datagen_config["d2_initial_weights"])
    cfg.DATASETS.TRAIN = ("marker_train",)
    cfg.DG_DATA_TYPE = datagen_config["data_type"]
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.SOLVER.IMS_PER_BATCH = datagen_config["ims_per_batch"]
    cfg.TEST.EVAL_PERIOD = datagen_config["eval_period"]
    cfg.SOLVER.STEPS = tuple(datagen_config["solver_steps"])
    cfg.SOLVER.MAX_ITER = datagen_config["solver_max_iter"]
    cfg.SOLVER.CHECKPOINT_PERIOD = datagen_config["checkpoint_period"]
    cfg.SOLVER.BASE_LR = datagen_config[
        "solver_base_lr"
    ]  # orig base lr divided by orig_ims_per_batch/current_ims_per_batch
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # 4 markers
    args.resume = args.resume or datagen_config["resume"]
    args.num_gpus = datagen_config["resume"]


def post_merging_config(cfg):
    if cfg.DG_DATA_TYPE not in ["real", "synt"]:
        raise ValueError("DG_DATA_TYPE must be 'real' or 'synt', while given type is: {}".format(cfg.DG_DATA_TYPE))
    if cfg.DG_TESTSET_NAME != "":
        cfg.DATASETS.TEST = ("marker_{}_{}".format(cfg.DG_TESTSET_NAME, cfg.DG_TESTSET_TYPE),)
        for testset_name in cfg.DATASETS.TEST:
            if testset_name not in DATASET_DICTS.keys():
                raise ValueError("Invalid test set name: {}".format(testset_name))
    else:
        cfg.DATASETS.TEST = ("marker_val_synt",)

    if cfg.DG_DATA_TYPE == "real":
        cfg.INPUT.MASK_FORMAT = "polygon"
    else:
        cfg.INPUT.MASK_FORMAT = "bitmask"


def setup(args):
    """hierarchy: args.opts > datagen_config > args.config_file"""
    cfg = get_datagen_cfg()
    # args.config_file
    if hasattr(args, "config_file") and args.config_file != "":
        cfg.merge_from_file(args.config_file)
    # datagen_config
    update_from_datagen_config(cfg, args)
    # args.opts
    cfg.merge_from_list(args.opts)
    post_merging_config(cfg)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def get_args():
    args = default_argument_parser().parse_args()
    datagen_config = get_datagen_config()
    args.num_gpus = datagen_config["num_gpus"]
    return args


"""
args = get_args()
cfg = setup(args)
"""
