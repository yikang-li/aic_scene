# Baseline for AI Challenger Scene Classification challenge

This is a project framework for AI Challenger Scene Classification Challenge with Python 3 and PyTorch. Densenet and Resnet Baseline methods are included. If you have any problems, please feel free to [email me](mailto:allen.li.thu@gmail.com). 

For more information about me, please explore my [homepage](http://www.cvboy.com/)

## Setup

### Requirement 

First install python 3 (we don't provide support for python 2). We advise you to install python 3 and pytorch with Anaconda:

- [python with anaconda](https://www.continuum.io/downloads)
- [pytorch with CUDA](http://pytorch.org)

```
conda create --name workspace python=3
source activate workspace
conda install pytorch torchvision cuda80 -c soumith
```

Then clone the repo (with the `--recursive` flag for submodules) and install the complementary requirements:

```
cd $HOME
git clone --recursive git@github.com:yikang-li/aic_scene.git
cd aic_scene
pip install -r requirements.txt
```

### Submodules

Our code has one external dependencies:

- [pertrained_models](https://github.com/Cadene/pretrained-models.pytorch) is used to initilialize the model with pretrained parameters


### Data 

Download the AI Challenger [Scene Classification dataset](https://challenger.ai/competition/scene/subject) and **Place all image folders and annotations files to ONE folder**. Then fill the corresponding paths to the   ```options/default.yaml```.
