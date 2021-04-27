import os
from detectron2.config import get_cfg
from detectron2 import model_zoo
from datetime import datetime
from detectron2.engine import default_setup
from detectron2.engine import default_argument_parser
import json


def get_datagen_cfg():
    """add placeholders to enable later loading of custom attributes
    from config_source in [args.config_file, datagen_config.json, args.opts]"""
    cfg = get_cfg()
    # cfg.DG_TRAIN_SET_PATHS = ""
    cfg.DG_TRAIN_SET_PATHS = tuple()
    cfg.DG_TRAIN_SET_INDS = tuple()  # ("0-750",)

    cfg.DG_TEST_SET_PATH = ""
    cfg.DG_TEST_SET_INDS = ""  # "0-200"

    cfg.DG_TRAINING_OUTPUT_DIR = ""
    cfg.DG_EVALUATION_OUTPUT_PATH = ""

    cfg.DG_MODE = ""  # train or test
    return cfg


def get_datagen_config():
    config_path = "settings/datagen_config.json"
    with open(config_path, "rb") as f:
        datagen_config = json.load(f)
    return datagen_config


def get_output_dir(cfg, datagen_config):
    # the case where no output dir given (default value)
    if cfg.OUTPUT_DIR == "./output":
        if "DG_TRAINING_OUTPUT_DIR" in datagen_config:
            return datagen_config["DG_TRAINING_OUTPUT_DIR"]
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
    cfg.DG_TRAIN_SET_PATHS = datagen_config["DG_TRAIN_SET_PATHS"]
    cfg.DG_TEST_SET_PATH = datagen_config["DG_TEST_SET_PATH"]
    cfg.DG_EVALUATION_OUTPUT_PATH = datagen_config["DG_EVALUATION_OUTPUT_PATH"]
    cfg.DG_TRAINING_OUTPUT_DIR = datagen_config["DG_TRAINING_OUTPUT_DIR"]
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        datagen_config["d2_initial_weights"]
    )
    cfg.DATASETS.TEST = ("marker_test",) if cfg.DG_MODE == "test" else tuple()
    cfg.SOLVER.IMS_PER_BATCH = datagen_config["ims_per_batch"]
    cfg.TEST.EVAL_PERIOD = datagen_config["eval_period"]
    cfg.SOLVER.STEPS = tuple(datagen_config["solver_steps"])
    cfg.SOLVER.MAX_ITER = datagen_config["solver_max_iter"]
    cfg.SOLVER.CHECKPOINT_PERIOD = datagen_config["checkpoint_period"]
    # orig base lr divided by orig_ims_per_batch/current_ims_per_batch
    cfg.SOLVER.BASE_LR = datagen_config["solver_base_lr"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # 4 markers
    args.resume = args.resume or datagen_config["resume"]
    args.num_gpus = datagen_config["num_gpus"]  # delete?


def post_merging_config(cfg):
    # handle mask format, all formats must be the same in case of multiple train sets
    set_paths = (
        cfg.DG_TRAIN_SET_PATHS if cfg.DG_MODE == "train" else (cfg.DG_TEST_SET_PATH, )
    )
    print(cfg.DG_TEST_SET_PATH)
    if set_paths[0] == "":
        mask_format = "bitmask"
    else:
        mask_formats = []
        for set_path in set_paths:
            annotations_path = os.path.join(set_path, "coco_annotations.json")
            mask_formats.append(get_mask_format(annotations_path))
        print(mask_formats)
        assert all([mask_formats[0] == mask_format for mask_format in mask_formats])
        mask_format = mask_formats[0]
    cfg.INPUT.MASK_FORMAT = mask_format

    cfg.OUTPUT_DIR = (
        cfg.DG_TRAINING_OUTPUT_DIR
        if cfg.DG_MODE == "train"
        else os.path.dirname(cfg.DG_EVALUATION_OUTPUT_PATH)
    )

    if cfg.DG_MODE == "train":
        cfg.DATASETS.TRAIN = tuple(
            "marker_train" + str(i) for i in range(len(cfg.DG_TRAIN_SET_PATHS))
        )
    else:
        cfg.DATASETS.TRAIN = tuple()


def get_mask_format(path_to_dataset) -> str:
    with open(path_to_dataset, "r") as f:
        dataset_dict = json.load(f)
    seg_instance = dataset_dict["annotations"][0]["segmentation"]
    if isinstance(seg_instance, list):
        return "polygon"
    elif isinstance(seg_instance, dict):
        return "bitmask"
    else:
        raise ValueError("Expected segmentation to be list or dictionary")


def setup(args, mode: str):
    """hierarchy: args.opts > datagen_config > args.config_file"""
    cfg = get_datagen_cfg()
    # args.config_file
    if hasattr(args, "config_file") and args.config_file != "":
        cfg.merge_from_file(args.config_file)
    # datagen_config
    assert mode in ["train", "test"]
    cfg.DG_MODE = mode
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
