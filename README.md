# RGBPoseTSA

This repository contains the FineDiving-Pose dataset and PyTorch implementation for RGBPoseTSA.

## Dataset
We used the LabelMe tool to label the human body bounding box for the FineDiving dataset. The annotation format is the same as the original format of the FineDiving dataset（ https://github.com/xujinglin/FineDiving ）Keep the same. 
Due to copyright reasons, we do not provide RGB image data of FineDiving dataset, but only bone point data and labels: 
BaiduNetDisk：FineDiving_RGBPose_annotation.pkl
website: https://pan.baidu.com/s/17vF3AGFzs7BJzoY80IuItA?pwd=dfps code: dfps

## Code for our work
### Requirement

- Python 3.7.9
- Pytorch 1.7.1
- torchvision 0.8.2
- timm 0.3.4
- torch_videovision
- mmcv

```
pip install git+https://github.com/hassony2/torch_videovision
```

### Pretrain Model

The Kinetics pretrained I3D downloaded from the reposity [kinetics_i3d_pytorch](https://github.com/hassony2/kinetics_i3d_pytorch/blob/master/model/model_rgb.pth)
