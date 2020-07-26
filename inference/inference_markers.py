import numpy as np
import cv2
import os
import shutil
import sys
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from settings.datagen_setup import setup, get_args
from data.prepare_datasets import prepare_datasets
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.catalog import Metadata
from data.prepare_datasets import THING_COLORS_BGR, THING_CLASSES
from utils.visualize import plot_side_by_side, plot_one_image


class VisualizerDG(Visualizer):
    def _jitter(self, x):
        return x


class MarkersInference(object):
    def __init__(self, cfg):
        self.predictor = DefaultPredictor(cfg)

    def predict_on_set(self, dataset_name, sample_size=4, show_plot=True, base_save_path=None):
        """
        perform inference on samples (or all images) from a given dataset name
        @param dataset_name: choose name from DATASET_DICTS located at data/prepare_datasets.py
        @param sample_size: how many images to show/plot
        @param show_plot: choose to show the plot or not (use False if you only want to save the plots as images)
        @param base_save_path: where to save the plots (if None --> it won't save the plots)
        """
        records, metadata = self._get_records(dataset_name, sample_size)
        for record in records:
            gt_image, pred_image = self._get_gt_and_pred_images(record, metadata)
            save_path = self._get_save_path(base_save_path, record)
            plot_side_by_side(
                imgs=[gt_image, pred_image],
                titles=["Ground Truth", "Prediction"],
                suptitle=os.path.basename(record["file_name"]),
                show_plot=show_plot,
                save_path=save_path,
            )

    def _get_records(self, dataset_name, sample_size):
        assert isinstance(sample_size, int) or sample_size == "all"
        record_dicts = DatasetCatalog.get(dataset_name)
        metadata = MetadataCatalog.get(dataset_name)
        if isinstance(sample_size, int):
            records = np.random.choice(record_dicts, size=sample_size, replace=False)
        else:
            records = record_dicts
        return records, metadata

    def _get_gt_and_pred_images(self, record, metadata):
        im = cv2.imread(record["file_name"])
        outputs = self.predictor(im)
        visualizer_object_gt = VisualizerDG(
            np.copy(im), metadata=metadata, scale=1, instance_mode=ColorMode.SEGMENTATION
        )
        gt_image = visualizer_object_gt.draw_dataset_dict(record).get_image()[:, :, ::-1]
        visualizer_object_pred = VisualizerDG(
            np.copy(im), metadata=metadata, scale=1, instance_mode=ColorMode.SEGMENTATION
        )
        pred_image = visualizer_object_pred.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[
            :, :, ::-1
        ]
        return gt_image, pred_image

    def _get_save_path(self, base_save_path, record):
        save_path = (
            None if base_save_path is None else os.path.join(base_save_path, os.path.basename(record["file_name"]))
        )
        return save_path

    def predict_on_video(self, video_path, show_plot=True, plot_two_images=True, output_path=None):
        """
        @param video_path: path to video 
        @param plot_two_images: plot_two_images: if True, will show raw image alongside image with predictions.
            if False, will only show the predictions.
        @param output_path: path to output directory
        """
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        count = 0
        while success:
            image_name = "frame{:05d}".format(count)
            save_path = os.path.join(output_path, image_name + ".png") if output_path is not None else None
            self.predict_on_image(
                image, title=image_name, show_plot=show_plot, plot_two_images=plot_two_images, save_path=save_path
            )
            success, image = vidcap.read()
            print("Read a new frame: ", success)
            count += 1

    def predict_on_image(self, img, title="", show_plot=True, save_path=None, plot_two_images=True):
        """

        @param img: numpy img to perform inference on, img can be also string with path to img
        @param title: the desired title of the figure
        @param show_plot: show plot or not
        @param save_path: where to save the the plotted image
        @param plot_two_images: if True, will show raw image alongside image with predictions.
            if False, will only show the predictions.
        """
        if isinstance(img, str):
            img = cv2.imread(img)
        outputs = self.predictor(img)
        metadata = self._get_metadata()
        self._plot_image_prediction(img, metadata, outputs, show_plot, plot_two_images, save_path, title)

    def _plot_image_prediction(self, img, metadata, outputs, show_plot, plot_two_images, save_path, title):
        visualizer_object = VisualizerDG(
            np.copy(img), metadata=metadata, scale=1, instance_mode=ColorMode.SEGMENTATION,
        )
        pred_image = visualizer_object.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]
        if plot_two_images:
            plot_side_by_side([img[:, :, ::-1], pred_image], ["Raw", "Prediction"], title, save_path)
        else:
            plot_one_image(pred_image, title, show_plot, save_path)

    def _get_metadata(self):
        metadata = Metadata()
        metadata.set(thing_colors=THING_COLORS_BGR)
        metadata.set(thing_classes=THING_CLASSES)
        return metadata


if __name__ == "__main__":
    DatasetCatalog.clear()
    args = get_args()
    cfg = setup(args)
    cfg.defrost()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
    cfg.freeze()
    prepare_datasets(cfg)
    marker_inference = MarkersInference(cfg)
    marker_inference.predict_on_set("marker_test_real")

    """
    python3 inference/inference_markers.py \
    DG_REAL_DATA_PATH path/to/real/data/dir  \  # can be set from datagen_config.json 
    DG_SYNT_DATA_PATH path/to/synt/data/dir \  # can be set from datagen_config.json
    MODEL.WEIGHTS model.pth \  # path to model_xxxx.pth - can be set from datagen_config.json
        
    # inference on set
    marker_inference.predict_on_set("marker_test_synt")

    # inference on image using image path
    marker_inference.predict_on_image(image_path)

    # inference on video using video path
    marker_inference.predict_on_video(video_path)
    """
