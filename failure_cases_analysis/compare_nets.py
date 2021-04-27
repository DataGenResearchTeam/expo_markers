from collections import defaultdict
import pandas as pd
from dataclasses import dataclass
import sys
from detectron2.engine import DefaultPredictor
from data.prepare_datasets import prepare_datasets, DATASET_NAMES
from failure_cases_analysis.calc_map_official import get_map_official
from inference.inference_markers import MarkersInference
from settings.datagen_setup import setup, get_args
from detectron2.data import DatasetCatalog, MetadataCatalog
import os
import pickle

from utils.visualize import plot_side_by_side

import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=1090, stdoutToServer=True, stderrToServer=True, suspend=False)


@dataclass
class Net:
    name: str
    path: str


def get_nets():
    nets = []
    net_real = Net("Real", "/mnt/system1/expo_paper2/training_logs/map_vs_size_india/expo_real_india_750_round002/model_0030999.pth")
    nets.append(net_real)
    net_synt = Net("Synt", "/mnt/system1/expo_paper2/training_logs/map_vs_size/expo_synt_v8_r3_4096_round004/model_0036999.pth")
    nets.append(net_synt)
    return nets


@dataclass
class TestSet:
    name: str
    path: str
    inds: str


def get_test_set():
    test_set = TestSet("Expo_Real_DGOffice",
                       "/datasets/Expo_Real_DataGen_Office_250/",
                       "0-20")
    return test_set


def get_cfg(net: Net, test_set: TestSet, th: float = 0):
    DatasetCatalog.clear()
    MetadataCatalog.clear()
    sys.argv = [
        "",
        "MODEL.WEIGHTS",
        net.path,
        "DG_TEST_SET_PATH",
        test_set.path,
        "DG_TEST_SET_INDS",
        test_set.inds,
    ]
    args = get_args()
    cfg = setup(args, "test")
    cfg.defrost()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = th
    cfg.freeze()
    return cfg


def compare_nets2(base_save_path):
    test_set = get_test_set()
    nets = get_nets()
    ddl = defaultdict(list)
    for net in nets:
        cfg = get_cfg(net, test_set)
        dataset_name = DATASET_NAMES["test"]
        prepare_datasets(cfg, [dataset_name])
        predictor = DefaultPredictor(cfg)
        record_dicts = DatasetCatalog.get(DATASET_NAMES["test"])
        metadata = MetadataCatalog.get(DATASET_NAMES["test"])
        records_lst, output_lst = [], []
        for k, record in enumerate(record_dicts[:10]):
            print(f"eval image num {k}")
            img = MarkersInference.load_img(record)
            output = predictor(img)
            records_lst.append(record)
            output_lst.append(output)

        coco_evaluations = []
        for record, output in zip(records_lst, output_lst):
            coco_eval = get_map_official([record], [output])
            coco_evaluations.append(coco_eval)
        coco_evaluations_path = os.path.join(base_save_path, "coco_evaluations.pickle")
        with open(coco_evaluations_path, "wb") as f:
            pickle.dump(coco_evaluations, f)

        coco_eval = get_map_official(records_lst, output_lst)
        coco_eval_path = os.path.join(base_save_path, "coco_eval.pickle")
        with open(coco_eval_path, "wb") as f:
            pickle.dump(coco_eval, f)


def compare_nets(base_save_path):
    test_set = get_test_set()
    nets = get_nets()
    ddl = defaultdict(list)
    for net in nets:
        cfg = get_cfg(net, test_set)
        dataset_name = DATASET_NAMES["test"]
        prepare_datasets(cfg, [dataset_name])
        predictor = DefaultPredictor(cfg)
        record_dicts = DatasetCatalog.get(DATASET_NAMES["test"])
        metadata = MetadataCatalog.get(DATASET_NAMES["test"])
        for record in record_dicts[:10]:
            img = MarkersInference.load_img(record)
            output = predictor(img)
            E = get_map_official([record], [output])
            gt_image, pred_image = MarkersInference.get_gt_and_pred_images(img, record, output, metadata)
            # dir_path = os.path.join(base_save_path, net.name)
            # os.makedirs(dir_path, exist_ok=True)
            # save_path = os.path.join(dir_path, os.path.basename(record["file_name"]))
            # assert save_path != record["file_name"]
            # fname = os.path.basename(record['file_name'])
            # sup = f"{fname}; {E.stats[0]}"
            # plot_side_by_side(
            #     imgs=[gt_image, pred_image],
            #     titles=["Ground Truth", "Prediction"],
            #     suptitle=sup,
            #     show_plot=False,
            #     save_path=save_path,
            # )

            ddl["net"].append(net.name)
            ddl["test_set"].append(test_set.name)
            ddl["image_id"].append(record["image_id"])
            ddl["metrics"].append(E.stats)
            ddl["gt_image_path"].append(None)


    df = pd.DataFrame(ddl)
    df_path = os.path.join(base_save_path, "evaluation_analysis.pickle")
    with open(df_path, "wb") as f:
        pickle.dump(df, f)


if __name__ == "__main__":
    base_save_path = "/mnt/system1/expo_paper2/failure_cases"
    os.makedirs(base_save_path, exist_ok=True)
    compare_nets2(base_save_path)


"""
docker run -it --rm \
--gpus device=0 --shm-size="8g" --env="DISPLAY" \
--net host \
-v /home/roey.ron/benchmarking/training_and_evaulation/detectron2/expo_markers:/expo_markers \
-v /mnt/system1/expo_paper2/:/mnt/system1/expo_paper2 \
-v /home/roey.ron:/home/roey.ron \
-v /home/roey.ron/directed_research_playground/datasets:/datasets \
-w /expo_markers \
-e PYTHONPATH=$PYTHONPATH:/expo_markers \
--name=detectron2_debug fastai_detectron2 python failure_cases_analysis/compare_nets.py
"""
