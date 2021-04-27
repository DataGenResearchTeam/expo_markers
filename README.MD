


<img src="https://i.imgur.com/w3l7y0U.jpg" width="250">  <br>

<font size=5>[Paper](https://arxiv.org/abs/2007.14354) | [Blog Post](https://www.datagen.tech/training-data-using-synthetic-expo-markers-to-train-an-object/)
</font> <br>
This repository will enable you to quickly perform visualization, inference, evaluation and training on our EXPO-HD Dataset with Mask R-CNN / Faster R-CNN using [Detectron2](https://github.com/facebookresearch/Detectron2/).
<br>
See our [Colab Notebook](https://colab.research.google.com/drive/1IjvwvA2xdufiIlTPtZgU6Ber03drbfaz?usp=sharing). <br><br>
[Visual Results](https://drive.google.com/drive/folders/1dvB5Gi71zaYvxjOmSVes0OkblGkPQRay?usp=sharing) of our trained netweork on our real image test set.
<br>

[comment]: <> (<img src="./expo_markers.gif" height=250>)
![](./expo_markers.gif)
# The EXPO-HD Dataset
EXPO-HD: Exact Object Perception using High Distraction Synthetic Data. <br>
The EXPO-HD Dataset is a dataset of Expo whiteboard markers for the purpose of <br> instance segmentation. 
The dataset contains three subsets (all include instances segmentation labels):  <br>
- Expo_Synt_V8 Photorealistic synthetic image dataset with 5000 images (1024x1024).
- Expo_Real_DGOffice Real image dataset with 250 images (used for validation and test).
- Expo_Real_India Real image dataset with 1000 images (used for training and comparison to our synthetic data).
<br><br>
<img src="https://i.imgur.com/7MKouv3.png" height=200>  <img src="https://i.imgur.com/iBeUCXV.jpg" height=200>  <img src="https://i.imgur.com/b0xrnBs.jpg" height=200>

# Our Code
## About detectron2
Our code is using detectron2 and is affected by detectron2's examples. <br>
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
using a config object which in our code is constructed from three sources with the 
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
DG_REAL_DATA_PATH path/to/real/data/dir  \
DG_SYNT_DATA_PATH path/to/synt/data/dir \
MODEL.WEIGHTS model.pth \
```

### Evaluation with Pre-trained Models

```
python3 evaluation/eval_markers.py \
MODEL.WEIGHTS /expo_markers/training_logs/expo_real_1000_0_750/model_final.pth \
DG_TEST_SET_PATH /expo_markers/expo_datasets/real_image_dataset/ \
DG_TEST_SET_INDS "('str0-50',)" \
DG_EVALUATION_OUTPUT_PATH /expo_markers/evaluation_logs/test_2/t666666666666666.pickle
```

### Training
```
python training/train_markers.py \
DG_TRAIN_SET_PATHS "('/path/to/dataset/dir',)" \
DG_TRAIN_SET_INDS "('0-750',)" \
DG_TRAINING_OUTPUT_DIR /path/to/weights/dir \
SOLVER.MAX_ITER 30000 \
SOLVER.STEPS '(25000, 27000)'
```


# License

(TO BE UPDATED) datagen_expo is released under [Apache 2.0 license](LICENSE).

# Citing

If you use our dataset or code in your research please use the following BibTeX entry.

```BibTeX
@ARTICLE {,
    author  = "Roey Ron and Gil Elbaz",
    title   = "EXPO-HD: Exact Object Perception using High Distraction Synthetic Data",
    journal = "arXiv",
    year    = "2020"
}
```