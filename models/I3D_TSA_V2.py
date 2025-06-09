import torch.nn as nn
import torch
from .I3D_backbone import I3D


class backbone(nn.Module):
    def __init__(self, action_number_choosing, **args):
        super(backbone, self).__init__()
        print('Using I3D_TSA_V2 backbone')
        self.backbone = I3D(**args)
        self.version = 'v2'
        self.action_number_choosing = action_number_choosing
        

    # use this when action_number_choosing = True
    def forward(self, video_12):
        if self.action_number_choosing:
            video_1, video_2 = video_12
            total_video = torch.cat((video_1, video_2), 0)
            total_feamap, total_feature = self.backbone(total_video)
            Nt, C, T, H, W = total_feamap.size()
            cls_feamap_12 = total_feamap
            total_feature = total_feature.reshape(len(total_video), 1, -1)
            total_feamap = total_feamap.reshape(len(total_video), 1, C, T, H, W)
            feature_1 = total_feature[:total_feature.shape[0] // 2]
            feature_2 = total_feature[total_feature.shape[0] // 2:]
            feamap_1 = total_feamap[:total_feamap.shape[0] // 2]
            feamap_2 = total_feamap[total_feamap.shape[0] // 2:]
            return feature_1, feature_2, feamap_1, feamap_2, cls_feamap_12
        else:
            total_video = video_12
            total_feamap, total_feature = self.backbone(total_video)
            Nt, C, T, H, W = total_feamap.size()
            cls_feamap = total_feamap
            total_feature = total_feature.reshape(len(total_video), 1, -1)
            total_feamap = total_feamap.reshape(len(total_video), 1, C, T, H, W)
            return total_feature, total_feamap, cls_feamap