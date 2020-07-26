


<img src="https://i.imgur.com/w3l7y0U.jpg" width="250">  <br>

This repository will enable you to quickly perform visualization, inference, evaluation and training on our EXPO-HD Dataset with MASK R-CNN using the [Detectron2](https://github.com/facebookresearch/Detectron2/).
<br>
See our [Colab Notebook](https://colab.research.google.com/drive/1IjvwvA2xdufiIlTPtZgU6Ber03drbfaz?usp=sharing). <br><br>
[Visual Results](https://drive.google.com/drive/folders/1dvB5Gi71zaYvxjOmSVes0OkblGkPQRay?usp=sharing) of our trained netweork on our real image test set.
<br>
<img src="expo_markers.gif?raw=true" height=250>
# The EXPO-HD Dataset
EXPO-HD: Exact Object Perception using High Distraction Synthetic Data. <br>
The EXPO-HD Dataset is a dataset of Expo whiteboard markers for the purpose of <br> instance segmentation. 
The dataset contains two subsets (both include instances segmentation labels):  <br>
- Photorealistic synthetic image dataset with 5000 images.
- Real image dataset with 200 images (used for validation and test). <br>
<img src="https://i.imgur.com/FaGXfox.png" height=300><img src="https://i.imgur.com/iBeUCXV.jpg" height=300>


# About detectron2
Our code is using detectron2 and is heavily affected by detectron2's examples.
<br>
[https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)
* It is powered by the [PyTorch](https://pytorch.org) deep learning framework.
* It [trains much faster](https://detectron2.readthedocs.io/notes/benchmarks.html).

Read their [blog post](https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/)
to see more demos and learn about detectron2.

## Installation

Install [detctron2](https://github.com/facebookresearch/detectron2)
using their instructions.


## Download the Expo Markers Dataset and Pretrained Weights
The data can be downloaded from our [blog post](https://www.datagen.tech/training-data-using-synthetic-expo-markers-to-train-an-object/).
## Quick Start
in detectron2, both the trainer and the predictor objects, are initialized
using a config file which in our code is constructed from three sources with the 
following hierarchy:
args.config_file < datagen_config < args.opts
1) args.config_file - optional (default='')
2) datagen_config - a json file located at settings/datagen_config.json 
   with our default parameters
3) args.opts - parameter settings from argparse. see settings/datagen_setup.py

[This Colab Notebook](https://colab.research.google.com/drive/1IjvwvA2xdufiIlTPtZgU6Ber03drbfaz) will show you 
how to use our code.
### Inference with Pre-trained Models

```
python3 inference/inference_markers.py \
DG_REAL_DATA_PATH path/to/real/data/dir  \  # can be set from datagen_config.json 
DG_SYNT_DATA_PATH path/to/synt/data/dir \  # can be set from datagen_config.json
MODEL.WEIGHTS model.pth \  # path to model_xxxx.pth - can be set from datagen_config.json
```

### Evaluation with Pre-trained Models

```
python3 evaluation/eval_markers.py \
DG_REAL_DATA_PATH path/to/real/data/dir  \  # can be set from datagen_config.json
DG_SYNT_DATA_PATH path/to/synt/data/dir \  # can be set from datagen_config.json
DG_TESTSET_TYPE type \  # [real, synt] (real_format=polygon, synt_format=bitmask)
DG_TESTSET_NAME name \ # [train, val, test] (test_size=100 or use DG_DATASET_SIZE, val_size=40, test_size=160)
MODEL.WEIGHTS model.pth \  # path to model_xxxx.pth
DG_EVAL_OUTPUT_PATH path/to/evaluation_report.pickle \  # where to save evaluation result (pickle file) ```
```

### Training
```
python3 training/train_markers.py \
DG_REAL_DATA_PATH path/to/real/data/dir  \  # can be set from datagen_config.json
DG_SYNT_DATA_PATH path/to/synt/data/dir \  # can be set from datagen_config.json
DG_DATASET_SIZE dataset_size \  # size of the synthetic training set
OUTPUT_DIR path/to/output/dir  # training directory location (where checkpoints and other files are being saved)
```


## License

(TO BE UPDATED) datagen_expo is released under [Apache 2.0 license](LICENSE).

## Citing

If you use our dataset or code in your research please use the following BibTeX entry.

```BibTeX
@ARTICLE {,
    author  = "Roey Ron and Gil Elbaz",
    title   = "EXPO-HD: Exact Object Perception using High Distraction Synthetic Data",
    journal = "arXiv",
    year    = "2020"
}
```