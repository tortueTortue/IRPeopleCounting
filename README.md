# Infrared People Counting

## Context

![alt text](https://github.com/tortueTortue/IRPeopleCounting/blob/master/doc/img/Dichotomy.png?raw=true)

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
python train.py --model_type 'ConvNeXt' --mae --head regression --dataset_root \Data\LLVIP
```

If you want to use mae pretraining, you have to train using the "--mae", after training the model will be saved in the /models/trained_models/. Then when training the classifier use, specify the path to the pretrained model to the "--mae_cp_path".

### Results & Measurements
All the training results and a measures are reported in the folder <b>/runs/[MODEL_NAME]_[DATETIME_OF_TRAINING]</b>

## Evaluation

- mode [String] : Which evaluation you want to make {<b>accuracy</b>|<b>localization</b>|<b>speed</b>}
- model_path [String] : Path to model.pt
- dataset_path [String] : Path to dataset
- split [String] : Which training split you want to make you evaluation on. By default it's test {<b>test</b>|<b>validation</b>|<b>train</b>}

### Count Accuracy

The count accuracy is the amount of predictions where the number of people predicted in the image is the exact real number divided by the total number of predictions.

Where Ŷ is predictions, $Y is Ground Truths,
$$ Count Accuracy = {\sum{Ŷ = Y} \over \sum{Ŷ}} $$

### Location

The localization results are measured using the mean Absolute Euclidean Distance (mAED). We calculate the mAED between the predicted coordinates and the ground truth for all images in the testing set. Both the $x$ and $y$ position coordinates are normalized within the range of 0 to 1. We give a 1 penalty for every wrong point predicted(or not predicted).

The mAED is computed as:

Where M is the amount of images in the dataset, N is the amount of points within the image, p and ^p respectively the closest ground truth and predicted points,
$$ mAED =  \frac{1}{M} \sum_{j=1}^{M}  \frac{1}{N_j} \sum_{i=1}^{N_j} ||p_{ij}-\hat{p}_{ij}||_{2}^{2} $$

### Speed

The speed benchmark is reported in FPS for both CPU and GPU.

If there is something missing, unclear or is not working in this repository, feel free to contact me and tell me.