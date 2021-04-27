import time
from detectron2.data.datasets import register_coco_instances
from copy import deepcopy
import json
import os
import re
import numpy as np
import contextlib
from collections import namedtuple
from typing import List


DATASET_NAMES = {"train": "marker_train", "test": "marker_test"}

THING_CLASSES = ["Red", "Green", "Blue", "Black"]
THING_COLORS_BGR = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (10, 10, 10)]


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def rgblst_2_bgrlst(rgb_lst):
    bgr_lst = [rgb[::-1] for rgb in rgb_lst]
    return bgr_lst


class SetManager(object):
    def __init__(self, json_path, shuffle=True):
        """
        The purpose of this class is to receive a path to a json file that represents a dataset and to enable the
        creation of distinct subsets (e.g. train, val, test) in the form of new json files.
        if shuffle is True, a constant shuffle will be applied, it's not relevant for the synthetic images, since
        they are completely random, but, for the real images it is more relevant, since there may be some non randomness
        in their order, for example, it's likely that two consecutive images were taken in the same location"""
        self._load_json_file(json_path)
        self._shuffle_indexes_if_needed(shuffle)

    def _shuffle_indexes_if_needed(self, shuffle):
        self.shuffle = shuffle
        num_of_images = len(self.coco_annotations_dict["images"])
        if self.shuffle:
            with temp_seed(0):
                self.inds = list(np.random.permutation(range(num_of_images)))
        else:
            self.inds = list(range(num_of_images))

    def _load_json_file(self, json_path):
        with open(json_path, "r") as f:
            self.coco_annotations_dict = json.load(f)

    def get_and_save_coco_annotations_subset(self, abs_range, output_json_path):
        new_coco_annotations_dict = self._get_dictionary_subset(abs_range)
        self._save_annotations_dict(output_json_path, new_coco_annotations_dict)
        return new_coco_annotations_dict

    def _get_dictionary_subset(self, abs_range):
        subset_inds = self.inds[abs_range[0] : abs_range[1]]
        new_coco_annotations_dict = deepcopy(self.coco_annotations_dict)
        new_coco_annotations_dict["images"] = [
            self.coco_annotations_dict["images"][ind] for ind in subset_inds
        ]
        img_ids = [e["id"] for e in new_coco_annotations_dict["images"]]
        new_coco_annotations_dict["annotations"] = list(
            filter(
                lambda ann: ann["image_id"] in img_ids,
                new_coco_annotations_dict["annotations"],
            )
        )
        return new_coco_annotations_dict

    def _save_annotations_dict(self, output_json_path, annotations_dict):
        with open(output_json_path, "w") as f:
            json.dump(annotations_dict, f)


class DatasetsPrepareTool(object):
    def __init__(
        self,
        cfg,
        dataset_names: List[str] = None,
        overwrite_existing_file=False,
    ):
        self.overwrite_existing_file = overwrite_existing_file
        self._assert_dataset_names_validity(dataset_names)
        self.dataset_names = dataset_names
        self.set_data_lst = self._get_set_data_lst(cfg)

    @staticmethod
    def _assert_dataset_names_validity(dataset_names):
        for dataset_name in dataset_names:
            print([e in dataset_name for e in DATASET_NAMES.values()])
            assert any(
                [e in dataset_name for e in DATASET_NAMES.values()]
            ), dataset_name

    def _get_set_data_lst(self, cfg):
        SetData = namedtuple("SetData", ["set_name", "set_path", "set_index_range_str"])
        set_data_lst = []

        train_sets_counter = 0
        for dataset_name in self.dataset_names:
            if "marker_train" in dataset_name:
                train_set_data = SetData(
                    set_name="marker_train" + str(train_sets_counter),
                    set_path=cfg.DG_TRAIN_SET_PATHS[train_sets_counter],
                    set_index_range_str=cfg.DG_TRAIN_SET_INDS[train_sets_counter],
                )
                train_sets_counter += 1
                set_data_lst.append(train_set_data)

        if "marker_test" in self.dataset_names:
            test_set_data = SetData(
                set_name="marker_test",
                set_path=cfg.DG_TEST_SET_PATH,
                set_index_range_str=cfg.DG_TEST_SET_INDS,
            )
            set_data_lst.append(test_set_data)
        return set_data_lst

    def prepare(self):
        for set_data in self.set_data_lst:
            self._prepare_dataset(set_data)
            print("DATASET WAS REGISTERED", set_data)

    def _prepare_dataset(self, set_data):
        orig_set_annotations_path = os.path.join(
            set_data.set_path, "coco_annotations.json"
        )
        output_json_path = os.path.join(
            set_data.set_path,
            "coco_annotations_" + set_data.set_index_range_str + ".json",
        )
        image_dir = os.path.join(set_data.set_path, "images")

        set_manager = SetManager(orig_set_annotations_path)

        if self._decide_whether_to_overwrite(output_json_path):
            set_range = self.convert_str_range_to_tuple(set_data.set_index_range_str)
            ping = time.time()
            set_manager.get_and_save_coco_annotations_subset(
                set_range, output_json_path
            )
            pong = time.time()
            print(
                f"TTTTTTT saved coco annotations: {output_json_path} took {pong - ping} seconds"
            )

        register_coco_instances(set_data.set_name, {}, output_json_path, image_dir)

    @staticmethod
    def convert_str_range_to_tuple(str_range):
        # '200-250' --> (200, 250)
        assert re.compile("\d+-\d+").fullmatch(str_range) is not None
        return tuple(int(e) for e in str_range.split("-"))

    def _decide_whether_to_overwrite(self, output_json_path):
        is_path_not_exists = not os.path.exists(output_json_path)
        return self.overwrite_existing_file or is_path_not_exists


def prepare_datasets(cfg, dataset_names=None):
    if dataset_names is not None:
        assert isinstance(dataset_names, list)
    datasets_prepare_tool = DatasetsPrepareTool(cfg, dataset_names)
    datasets_prepare_tool.prepare()
