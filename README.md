# RGBPoseTSA

This repository contains the FineDiving-Pose dataset and PyTorch implementation for RGBPoseTSA.

## Dataset
We used the LabelMe tool to label the human body bounding box for the FineDiving dataset. The annotation format is the same as the original format of the FineDiving dataset（ https://github.com/xujinglin/FineDiving ）Keep the same. 
Due to copyright reasons, we do not provide RGB image data of FineDiving dataset, but only bone point data and labels: 
BaiduNetDisk：FineDiving_RGBPose_annotation.pkl
website: https://pan.baidu.com/s/17vF3AGFzs7BJzoY80IuItA?pwd=dfps code: dfps

| Field Name    | Type   | Description                          | Field Name             | Type  | Description                         |
| ------------- | ------ | ------------------------------------ | ---------------------- | ----- | ----------------------------------- |
| `action_type` | string | Description of the action type.      | `sub-action_types`     | dict  | Description of the sub-action type. |
| `label`       | int    | action type in int format.           | `judge_scores`         | list  | Judge scores.                       |
| `dive_score`  | float  | Diving score of the action instance. | `frames_labels`        | array | Step-level labels of the frames.    |
| `difficulty`  | float  | Difficulty of the action type.       | `steps_transit_frames` | array | Frame index of step transitions.    |
| `img_shape`   | tuple  | Length and width of the picture.     | `total_frames`         | int   | Frames of original video(images).   |
| `gtboxes`     | array  | Human bounding box coordinates.      | `keypoint`             | array   | Human skeleton data extracted through HRNet.  |
| `keypoint_score`  | array  | Human bounding box coordinates. 

Note: 
- The shape of gtboxes is [T, N, C]. T represents the number of frames, N represents the number of people, C represents the channel, and C defaults to 4, corresponding to (x1, y1, x2, y2), that is, the coordinates of the upper left and lower right corners of the bounding box. 
- The shape of the keypoint is [M, T, V, C]. M represents the number of people, T represents the time, V represents the joint point, C represents the channel, and C is 2 by default, which corresponds to the coordinates of the key point. 
- The shape of keypoint_score is [M, T, V]. M represents the number of people, T represents the time, and V represents the joint point. Indicates the confidence socre of key points estimated by HRNet.



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
