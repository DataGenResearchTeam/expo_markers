from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from copy import deepcopy
import json
import os
import numpy as np
import contextlib

DATASET_DICTS = {
    "marker_train": {"range": None, "type": "synt"},
    "marker_val_synt": {"range": [4600, 4800], "type": "synt"},
    "marker_val_real": {"range": [0, 40], "type": "real"},
    "marker_test_synt": {"range": [4800, 5000], "type": "synt"},
    "marker_test_real": {"range": [40, 200], "type": "real"},
}

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
    bgr_lst = [(rgb[2], rgb[1], rgb[0]) for rgb in rgb_lst]
    return bgr_lst


class SetManager(object):
    def __init__(self, json_path, shuffle=False):
        """
        The purpose of this class is to receive a path to a json file that represents a dataset and to enable the
        creation of distinct subsets (e.g. train, val, test) in the form of new json files.
        if shuffle is True, a constant shuffle will be applied, it's not relevant for the synthetic images, since
        they are completely random, but, for the real images it's more relevant, since there may be some non randomness
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
        new_coco_annotations_dict["images"] = [self.coco_annotations_dict["images"][ind] for ind in subset_inds]
        img_ids = [e["id"] for e in new_coco_annotations_dict["images"]]
        new_coco_annotations_dict["annotations"] = list(
            filter(lambda ann: ann["image_id"] in img_ids, new_coco_annotations_dict["annotations"])
        )
        return new_coco_annotations_dict

    def _save_annotations_dict(self, output_json_path, annotations_dict):
        with open(output_json_path, "w") as f:
            json.dump(annotations_dict, f)


class DatasetsPrepareTool(object):
    def __init__(self, cfg, overwrite_existing_file=True):
        self.overwrite_existing_file = overwrite_existing_file
        self._init_parameters(cfg)
        self._init_setmanagers()

    def _init_parameters(self, cfg):
        self.synt_data_path = cfg.DG_SYNT_DATA_PATH
        self.real_data_path = cfg.DG_REAL_DATA_PATH
        self.train_set_size = cfg.DG_DATASET_SIZE

    def _init_setmanagers(self):
        self.set_manager_synt = SetManager(os.path.join(self.synt_data_path, "coco_annotations.json"))
        self.set_manager_real = SetManager(os.path.join(self.real_data_path, "coco_annotations.json"), shuffle=True)

    def prepare(self, dataset_names=None):
        dataset_names = self._get_dataset_names(dataset_names)
        for dataset_name in dataset_names:
            self._prepare_dataset_by_name(dataset_name)

    def _get_dataset_names(self, dataset_names):
        if dataset_names is None:
            dataset_names = DATASET_DICTS.keys()
        else:
            assert isinstance(dataset_names, list)
            assert len(set(dataset_names) - set(DATASET_DICTS.keys())) == 0
        return dataset_names

    def _prepare_dataset_by_name(self, dataset_name):
        absolute_range, set_manager, data_path, output_json_path, image_dir = self._get_dataset_attributes(dataset_name)
        if self._decide_whether_to_overwrite(output_json_path, dataset_name):
            set_manager.get_and_save_coco_annotations_subset(absolute_range, output_json_path)
        register_coco_instances(dataset_name, {}, output_json_path, image_dir)
        MetadataCatalog.get(dataset_name).set(thing_colors=THING_COLORS_BGR)

    def _get_dataset_attributes(self, dataset_name):
        dataset_dict = DATASET_DICTS[dataset_name]
        absolute_range = [0, self.train_set_size] if dataset_name == "marker_train" else dataset_dict["range"]
        set_manager = self.set_manager_synt if dataset_dict["type"] == "synt" else self.set_manager_real
        data_path = self.synt_data_path if dataset_dict["type"] == "synt" else self.real_data_path
        output_json_path = os.path.join(data_path, "coco_annotations_" + dataset_name + ".json")
        image_dir = os.path.join(data_path, "images")
        return absolute_range, set_manager, data_path, output_json_path, image_dir

    def _decide_whether_to_overwrite(self, output_json_path, dataset_name):
        is_path_not_exists = not os.path.exists(output_json_path)
        is_marker_train = dataset_name == "marker_train"
        return self.overwrite_existing_file or is_path_not_exists or is_marker_train


def prepare_datasets(cfg):
    datasets_prepare_tool = DatasetsPrepareTool(cfg)
    datasets_prepare_tool.prepare()
