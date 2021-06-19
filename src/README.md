# Training and Inference Instructions

## Before Tranining and Inference

To generate the distance maps, you need to run `Gen_BDist_Map.ipynb`, see [details](./Gen_BDist_Map.ipynb)

To generate the training and test patches, you need to run `Extract_Patches_Own_data.ipynb`, see [details](./Extract_Patches_Own_data.ipynb)

Finish these two steps, the dataset has been already to use.

## Choosing the network

The model to use and the selection of other hyperparameters is selected in `config.py`. The models available are:
- CHR-Net: `model/hcnet.py`
- HoVer-Net: `model/graph.py`
- DIST: `model/dist.py`
- Micro-Net: `model/micronet.py`
- DCAN: `model/dcan.py`
- SegNet: `model/segnet.py`
- U-Net: `model/unet.py`
- FCN8: `model/fcn8.py`

To use the above models, modify `mode` and `self.model_type` in `config.py` as follows:

- CHR-Net: `mode='hcnet'` , `self.model_type=hcnet`
- HoVer-Net: `mode='hover'` , `self.model_type=np_hv`
- DIST: `mode='other'` , `self.model_type=dist`
- Micro-Net: `mode='other'` , `self.model_type=micronet`
- DCAN: `mode='other'` , `self.model_type=dcan`
- SegNet: `mode='other'` , `self.model_type=segnet`
- U-Net: `mode='other'` , `self.model_type=unet`
- FCN8: `mode='other'` , `self.model_type=fcn8`

## Modifying Hyperparameters

To modify hyperparameters, refer to `opt/`. For CHR-Net, modify the script `opt/other.py` (`hcnet={...}`).

## Augmentation

To modify the augmentation pipeline, refer to `get_train_augmentors()` in `config.py`. Refer to [this webpage](https://tensorpack.readthedocs.io/modules/dataflow.imgaug.html)        for information on how to modify the augmentation parameters.

## Training

To train the network:

Run the `Trainning.ipynb`.

Set the `gpus`, which denotes which GPU will be used for training.

Before training, set in `config.py` and `opt/other.py`:
- path to pretrained weights ResNet34. Download the weights [here](https://nextcloud.chenli.group/index.php/s/wgNMPLa8ZGBCAtP).
- path to the data directories
- path where checkpoints will be saved

## Inference

To do the inference:

Run the `Predicting.ipynb`.

Set the `gpus`, which denotes which GPU will be used for inference.

Before running inference, set in `config.py`:
- path where the output will be saved
- path to data root directories
- path to model checkpoint


