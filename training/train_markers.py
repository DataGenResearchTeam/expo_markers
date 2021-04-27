import os
from settings.datagen_setup import setup, get_args
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator
from data.prepare_datasets import DatasetsPrepareTool


class DataGenTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def main(args):
    cfg = setup(args, mode="train")
    datasets_prepare_tool = DatasetsPrepareTool(cfg, cfg.DATASETS.TRAIN)
    datasets_prepare_tool.prepare()
    print(cfg)
    trainer = DataGenTrainer(cfg)
    trainer.resume_or_load(args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = get_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

"""
docker run -id --rm --gpus all --shm-size="8g" --env="DISPLAY" \
-p 6006:6006 -p 8888:8888 \
-v /home/roey.ron/benchmarking/training_and_evaulation/detectron2/expo_markers:/expo_markers \
-v /mnt/system1/expo_paper2/:/mnt/system1/expo_paper2 \
-v /home/roey.ron:/home/roey.ron \
-v /home/roey.ron/expo_markers_experiments/:/experiments \
-v /home/roey.ron/directed_research_playground/datasets:/datasets \
-w /expo_markers \
-e PYTHONPATH=$PYTHONPATH:/expo_markers \
--name=detectron2 fastai_detectron2

docker exec -it detectron2 bash


jupyter notebook /expo_markers --ip=0.0.0.0 --port=8889 --allow-root --no-browser

tensorboard --logdir /compare_mask_rcnn_to_rcnn/training_logs --host 0.0.0.0



docker run -it --rm --gpus all --shm-size="8g" --env="DISPLAY" \
-v /home/roey.ron/benchmarking/training_and_evaulation/detectron2/expo_markers:/expo_markers \
-v /home/roey.ron/expo_markers_experiments/:/experiments \
-v /mnt/system1/expo_paper2/:/mnt/system1/expo_paper2 \
-v /home/roey.ron/directed_research_playground/datasets:/datasets \
-w /expo_markers \
-e PYTHONPATH=$PYTHONPATH:/expo_markers \
--name=detectron2_debug fastai_detectron2 \
bash


python training/train_markers.py \
DG_TRAIN_SET_PATHS "('/datasets/Expo_Synt_V8_R3', '/datasets/Expo_Real_India')" \
DG_TRAIN_SET_INDS "('0-750', '0-50')" \
DG_TRAINING_OUTPUT_DIR /mnt/system1/expo_paper2/test_system1 \
SOLVER.MAX_ITER 30000 \
SOLVER.STEPS '(25000, 27000)'
"""
