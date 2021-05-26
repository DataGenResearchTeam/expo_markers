


<img src="https://i.imgur.com/w3l7y0U.jpg" width="250">  <br>

<font size=5>[Paper](https://arxiv.org/abs/2007.14354) | [Blog Post](https://www.datagen.tech/training-data-using-synthetic-expo-markers-to-train-an-object/)
</font> <br>
This repository will enable you to quickly perform visualization, inference, evaluation and training on our Expo Markers Dataset with Mask R-CNN / Faster R-CNN using [Detectron2](https://github.com/facebookresearch/Detectron2/).
<br>
See our [Colab Notebook](https://colab.research.google.com/drive/1IjvwvA2xdufiIlTPtZgU6Ber03drbfaz?usp=sharing). <br><br>
[Visual Results](https://drive.google.com/drive/folders/1dvB5Gi71zaYvxjOmSVes0OkblGkPQRay?usp=sharing) of our trained netweork on our real image test set.
<br>

[comment]: <> (<img src="./expo_markers.gif" height=250>)
![](./expo_markers.gif)
# The EXPO Markers Dataset
The EXPO Markers Dataset is a dataset of Expo whiteboard markers for the purpose of instance segmentation and object detection. 
The dataset contains three subsets (all include instances segmentation labels):  <br>
- Expo_Synt_V8 Photorealistic synthetic image dataset with 5000 images (1024x1024).
- Expo_Real_DGOffice Real image dataset with 250 images (used for validation and test).
- Expo_Real_India Real image dataset with 1000 images (used for training and comparison to our synthetic data).
<br><br>
<img src="https://i.imgur.com/7MKouv3.png" height=180>  <img src="https://i.imgur.com/iBeUCXV.jpg" height=180>  <img src="https://i.imgur.com/b0xrnBs.jpg" height=180>

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
args.config_file < datagen_config < args.opts.
1) args.config_file - optional (default='').
2) datagen_config - a json file located at settings/datagen_config.json
   with our default parameters.
3) args.opts - parameter settings from argparse. see settings/datagen_setup.py.

[This Colab Notebook](https://colab.research.google.com/drive/1IjvwvA2xdufiIlTPtZgU6Ber03drbfaz) will show you 
how to use our code.
### Inference with Pre-Trained Models

```
python inference/inference_markers.py 
```

### Evaluation with Pre-Trained Models

```
python evaluation/eval_markers.py \
MODEL.WEIGHTS /path/to/weights/model.pth \
DG_TEST_SET_PATH /expo_markers/expo_datasets/real_image_dataset/ \
DG_TEST_SET_INDS "('0-50',)" \
DG_EVALUATION_OUTPUT_PATH /output/path/eval.pickle
```

### Training
```
python training/train_markers.py \
DG_TRAIN_SET_PATHS "('/path/to/dataset/dir/',)" \
DG_TRAIN_SET_INDS "('0-750',)" \
DG_TRAINING_OUTPUT_DIR /path/to/output/weights/dir/ \
SOLVER.MAX_ITER 30000 \
SOLVER.STEPS '(25000, 27000)'
```


# License

The Datagen expo dataset is released under [Non-Commercial Data License Agreement](https://www.datagen.tech/data-license-agreement/).

# Citing

If you use our dataset or code in your research please use the following BibTeX entry.

```BibTeX
@misc{ron2021detection,
      title={Detection and Segmentation of Custom Objects using High Distraction Photorealistic Synthetic Data}, 
      author={Roey Ron and Gil Elbaz},
      year={2021},
      eprint={2007.14354},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```