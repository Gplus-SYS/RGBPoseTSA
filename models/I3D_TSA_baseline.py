import torch.nn as nn
import torch
from .I3D_backbone import I3D

class backbone(nn.Module):
    def __init__(self, action_number_choosing, **args):
        super(backbone, self).__init__()
        print('Using I3D_TSA backbone')
        self.backbone = I3D(**args)
        self.version = 'v1'
        self.action_number_choosing = action_number_choosing
    
    def forward(self, video_12):
        if self.action_number_choosing:
            video_1, video_2 = video_12
            total_video = torch.cat((video_1, video_2), 0)
            start_idx = list(range(0, 90, 10))
            video_pack = torch.cat([total_video[:, :, i: i + 16] for i in start_idx])

            total_feamap, total_feature = self.backbone(video_pack)# shape = [144, 1024, 2, 4, 4], [144, 1024, 1, 1, 1]
            Nt, C, T, H, W = total_feamap.size()

            total_feature = total_feature.reshape(len(start_idx), len(total_video), -1).transpose(0, 1) # shape:[16, 9, 1024],[2*batchsize,length,channel]
            total_feamap = total_feamap.reshape(len(start_idx), len(total_video), C, T, H, W).transpose(0, 1) # shape:[16, 9, 1024, 2, 4, 4]

            Batch_size, Clips, Channels, Times, Height, Width = total_feamap.size()
            cls_feamap_12 = total_feamap.reshape(Batch_size, Channels, Times*Clips, Height, Width)

            feature_1 = total_feature[:total_feature.shape[0] // 2]
            feature_2 = total_feature[total_feature.shape[0] // 2:]

            feamap_1 = total_feamap[:total_feamap.shape[0] // 2]
            feamap_2 = total_feamap[total_feamap.shape[0] // 2:]

            return feature_1, feature_2, feamap_1, feamap_2, cls_feamap_12
        else:
            start_idx = list(range(0, 90, 10))
            video_pack = torch.cat([video_12[:, :, i: i + 16] for i in start_idx])

            total_feamap, total_feature = self.backbone(
                video_pack)  # shape = [144, 1024, 2, 4, 4], [144, 1024, 1, 1, 1]
            Nt, C, T, H, W = total_feamap.size()

            total_feature = total_feature.reshape(len(start_idx), len(video_12), -1).transpose(0,
                                                                                              1)  # shape:[16, 9, 1024],[2*batchsize,length,channel]
            total_feamap = total_feamap.reshape(len(start_idx), len(video_12), C, T, H, W).transpose(0,
                                                                                                    1)  # shape:[16, 9, 1024, 2, 4, 4]

            Batch_size, Clips, Channels, Times, Height, Width = total_feamap.size()
            cls_feamap = total_feamap.reshape(Batch_size, Channels, Times * Clips, Height, Width)

            return total_feature, total_feamap, cls_feamap
