from settings.datagen_setup import setup, get_args
from detectron2.engine import launch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import verify_results
import pickle
import detectron2.utils.comm as comm
from data.prepare_datasets import DatasetsPrepareTool
from training.train_markers import DataGenTrainer


def main(args):
    cfg = setup(args, mode="test")
    datasets_prepare_tool = DatasetsPrepareTool(cfg, cfg.DATASETS.TEST)
    datasets_prepare_tool.prepare()
    model = DataGenTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    res = DataGenTrainer.test(cfg, model)
    if comm.is_main_process():
        verify_results(cfg, res)
        with open(cfg.DG_EVALUATION_OUTPUT_PATH, "wb") as f:
            pickle.dump(res, f)
        print("evaluation report was saved to {}".format(cfg.DG_EVALUATION_OUTPUT_PATH))
    return res


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
