from detectron2.data import transforms as T
import copy
import detectron2.data.detection_utils as utils
import torch
import numpy as np


def mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    image, transforms = T.apply_transform_gens(
        [
            T.RandomFlip(prob=0.20, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.20, horizontal=False, vertical=True),
            T.RandomApply(
                T.RandomBrightness(intensity_min=0.75, intensity_max=1.25), prob=0.20
            ),
            T.RandomApply(
                T.RandomContrast(intensity_min=0.76, intensity_max=1.25), prob=0.20
            ),
            T.RandomApply(
                T.RandomCrop(crop_type="relative_range", crop_size=(0.8, 0.8)),
                prob=0.20,
            ),
            T.RandomApply(
                T.RandomSaturation(intensity_min=0.75, intensity_max=1.25), prob=0.20
            ),
            T.RandomApply(T.Resize(shape=(800, 800)), prob=0.20),
            T.RandomApply(
                T.RandomRotation(
                    angle=[-30, 30],
                    expand=True,
                    center=None,
                    sample_style="range",
                    interp=None,
                ),
                prob=0.20,
            ),
        ],
        image,
    )

    dataset_dict["image"] = torch.as_tensor(
        np.ascontiguousarray(image.transpose(2, 0, 1))
    )

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict
