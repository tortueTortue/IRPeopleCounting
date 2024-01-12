# Infrared People Counting

## Context

![alt text](https://github.com/tortueTortue/IRPeopleCounting/blob/master/Dichotomy.png?raw=true)

This is an implementation of the people counting image-level models from the paper : Evaluating Supervision Levels Trade-Offs for Infrared-Based People Counting.
https://arxiv.org/pdf/2311.11974.pdf

## Installation
```console
pip install -r requirements.txt
```

## Dataset
The dataset used needs to be structured into the following way :<br/>
<b>/<i>dataset_name</i>/<i>split</i>[train | test]/<i>people_count</i>[0 ... MAX_PEOPLE_COUNT]/<i>image_name.img_ext</i>[jpg | png]</b>

Here is an example of the dataset folder structure:

/LLVIP <br/>
&nbsp;&nbsp;&nbsp;&nbsp;/train <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/0 <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/im_00.png <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/im_01.jpg <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/im_02.jpg <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/1 <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/im_67.jpg <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;/im_87.png <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ... <br/>
&nbsp;&nbsp;&nbsp;&nbsp;/test <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;... <br/>

## Training

### Command & Arguments

- mae [Boolean] : Whether to use Masked Autoencoder pretraining or not.
- model_type [String] : Whether to use ViT or ConvNeXt.
- pretrained [Boolean] : Whether to use pretrained weights from ImageNet or not.
- head [String] : Whether to use a classification head or regression head. (classification | regression)
- dataset_root [String] : Dataset Root path
- mae_cp_path [String] : Checkpoint path to a previous Masked Autoencoder Pretraining.
- sub_ratio [Float] : Ratio of the training data used for training. Mostly for experimental purposes.
- small [Boolean] : Whether to use the small version or normal of the model.

```console
python train.py --model_type 'ConvNeXt' --mae true --head regression --dataset_root \Data\LLVIP
```

### Results & Measurements
All the training results and a measures are reported in the folder <b>/runs/[MODEL_NAME]_[DATETIME_OF_TRAINING]</b>

## Evaluation

- model_path [String] : 
- mode [String] : 

### Count Accuracy

### Location

### Speed