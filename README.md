# Semantic Segmentation fine-tuning using DeepLabv3 ResNet50 on Cityscapes dataset
### Introduction
This project entails the fine-tuning of the [popular DeepLabv3 PyTorch implementation](https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.deeplabv3_resnet50.html) using the ResNet50 backbone. It makes use of the fine Cityscapes labels for more realistic real-life scenarios. With the parameters defined in the training notebook, the model achieved **~62%** mIoU performance. See [report.pdf](report.pdf) for the paper associated with this project.

## Setup
The model was conceptualized using 1x H100 80GB GPU, while the final training run was executed on 8x A100 40GB GPUs.

### Setup: How to use
**Pre-requisites:**

```
Python 3.10 or higher
pip
```

Requirements

```python
os
torch
Albumentations
torchmetrics
tqdm
matplotlib
numpy
cityscapesscripts
```
## Tutorial
### Tutorial: Dataset preparation
Step 1: Download the dataset using [download_dataset.ipynb](src/download_dataset.ipynb). You require a Cityspaces account which can be created [here](https://www.cityscapes-dataset.com/), in which you will be prompted your login information when running the notebook.

Step 2: After successfully downloading the .zip files in the [data](data) folder, you must extract the zip files inside that folder. You will be left with [data/gtFine](data/gtFine) and [data/leftImg8bit](data/leftImg8bit) and their respect train, val and test sub-folders.

Step 3: The last step in preparing the data is to run the second part of [download_dataset.ipynb](src/download_dataset.ipynb). This will generate the training label IDs which are a necessity for training as they are not directly provided in the dataset.

*Note: The test dataset cannot be used as all the reference images in that dataset are blacked out.*

### Tutorial: How to fine-tune
*The code is untested on CPU applications, and is optimized for CUDA applications.*

The prototyping notebook in [cityscapes.ipynb](src/cityscapes.ipynb) can be run on a 1x H100 80GB GPU. Adjust the image resolution in the Albumentations transforms as needed, as well as the batch size and workers.

The final training notebook in [cityscapes_parallel.ipynb](src/cityscapes_parallel.ipynb) can be run on 8x A100 40GB GPUs. Note the increased resolution of the inputs, as well as the increased batch sizes. Note that the batch size needs to be increased proportionally to the amount of GPUs for training.

### Tutorial: Inference
Baseline: Before training, you may run [baseline_inference.ipynb](src/baseline_inference.ipynb) to get an idea of the model's performance on the Cityscape's dataset using only its pre-trained weights. With the current performance, the pre-trained model is simply un-usable. After only one fine-tuning the model becomes drastically better, so it is not recommended to make conclusions simply based on the baseline.

Fine-tuned: After training, if you save the model checkpoints in [src/checkpoints](src/checkpoints), you can then do an evaluation pass using [checkpoint_inference.ipynb](src/checkpoint_inference.ipynb) to get an idea of the final accuracy. This notebook simply loads the model and the checkpoint, and then does a forward pass on all the validation dataset.

Visualization: If you wish to visualize the model's output segmentation, you can use *visualize_segmentation* from the main training notebooks. You have to specify a pre-processed dataset and the index of the image you want to visualize.