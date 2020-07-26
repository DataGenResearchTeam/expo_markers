import os
from settings.datagen_setup import setup, get_args
from detectron2.engine import DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator
from data.prepare_datasets import prepare_datasets, DatasetsPrepareTool


class DataGenTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)


def main(args):
    cfg = setup(args)
    datasets_prepare_tool = DatasetsPrepareTool(cfg)
    datasets_prepare_tool.prepare(list(set(cfg.DATASETS.TRAIN + cfg.DATASETS.TEST)))
    trainer = DataGenTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
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
python3 training/train_markers.py \
DG_REAL_DATA_PATH path/to/real/data/dir  \  # can be set from datagen_config.json
DG_SYNT_DATA_PATH path/to/synt/data/dir \  # can be set from datagen_config.json
DG_DATASET_SIZE dataset_size \  # size of the synthetic training set
OUTPUT_DIR path/to/output/dir  # training directory location (where checkpoints and other files are being saved)
"""
