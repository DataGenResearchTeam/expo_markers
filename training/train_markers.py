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
