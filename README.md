# CNN–GRU for GHI prediction

<!-- ------------------------------------------------------------------------------ -->
## Introduction .
Spatial and temporal feature is crucial for time-series forecasting. ResNet-18 can extract the image feature while GRU can extract temporal feature.

Our method:
1)  We use ResNet-18 to extract the image feature in different. 
2)  We use GRU to extract the temporal feature by using the sliding windows. 

<!-- ------------------------------------------------------------------------------ -->
## File structure and file description
**Dirs:**
- pre_trained: The model that we have trained(will be added later)
- model: The implementation of our model
- save: A directory that saves the model dict, prediction result and train_val loss.
- result: save the prediction result.

**Files:**
- main.py: The core file which achieve the training function.(The start of the program)
- dataset.py: rewrite the Dataset class so that achieve the data read function.
- test.py: The core file which achieve the test(model evaluation) function.
- config.py: A configuration file which set the all configurations including 'train_data_path','lr','epoch','batch_size' and so on.
- irr_data.csv: The radiation data corresponding to the image data. 
- environment.yaml: The environment dependency of our experiments.

Attention: Notes have been attached in the file for code readability.

<!-- ------------------------------------------------------------------------------ -->
## Results 
The evaluation index is: MAR,RMSE and CORR,
We test the model in a dataset which contains 3201 images.

<!-- -------------------------------------------------------- -->
## Run 

0. Requirements:
    * Install python3.6
    * Install csv,numpy,pandas 
    * Install [pytorch](https://pytorch.org/) (tested on Release 1.6)

Quick install: The environment dependencies is saved as 'environment.yaml'. 
You can clone the same environment by using the following code:
 ` conda env create -f environment.yaml`

1. Training:
    * Prepare training images split.
    * Modify [config.py](config.py) to set path to data, iterations, and other parameters.
    * Run `python train.py -train_path [training data path] -lr [learning_rate] -batch_size [batch_size] -gpu [the gpu id for training] --save [the dir name for saving the result] ` \

    * For example, `python main.py --epoch 100 --lr  0.001 -batch_size 32 -gpu 0 --save 03-28-09:10 --save_name prediction  `

2. Testing:
    * Run `python test.py --test_path --epoch --lr --save --save_name `. 
    * For example, `python test.py --epoch 100 --lr  0.001 -batch_size 32 -gpu 0 --save 03-28-09:10 --save_name prediction  `
**Attention**: The 'save' parameters in Testing should be the sanme in Training, as model dict is saved in the 'save' dir.


<!-- ------------------------------------------------------------------- -->
## Pretrained models
We will add the pretrained models in the future.

<!-- ------------------------------------------------------------------- -->

### License
Licensed under an MIT license.

### Contact
yuzhang0574@gmail.com