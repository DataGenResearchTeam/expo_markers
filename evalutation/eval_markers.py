from settings.datagen_setup import setup, get_args
from detectron2.engine import launch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import verify_results
import pickle
import detectron2.utils.comm as comm
from data.prepare_datasets import prepare_datasets, DatasetsPrepareTool
from training.train_markers import DataGenTrainer


def main(args):
    cfg = setup(args)
    datasets_prepare_tool = DatasetsPrepareTool(cfg)
    datasets_prepare_tool.prepare(list(cfg.DATASETS.TEST))
    model = DataGenTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
    res = DataGenTrainer.test(cfg, model)
    if comm.is_main_process():
        verify_results(cfg, res)
        with open(cfg.DG_EVAL_OUTPUT_PATH, "wb") as f:
            pickle.dump(res, f)
        print("evaluation report was saved to {}".format(cfg.DG_EVAL_OUTPUT_PATH))
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

""" 
python3 evaluation/eval_markers.py \
DG_REAL_DATA_PATH path/to/real/data/dir  \  # can be set from datagen_config.json
DG_SYNT_DATA_PATH path/to/synt/data/dir \  # can be set from datagen_config.json
DG_TESTSET_TYPE type \  # [real, synt] (real_format=polygon, synt_format=bitmask)
DG_TESTSET_NAME name \ # [train, val, test] (test_size=100 or use DG_DATASET_SIZE, val_size=40, test_size=160)
MODEL.WEIGHTS model.pth \  # path to model_xxxx.pth
DG_EVAL_OUTPUT_PATH path/to/evaluation_report.pickle \  # where to save evaluation result (pickle file) 
"""
