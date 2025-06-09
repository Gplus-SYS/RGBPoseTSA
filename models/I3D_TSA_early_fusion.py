import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import torch.nn as nn
import torch
from I3D_backbone import I3D
from utils.misc import import_class
from weight_init import kaiming_init,constant_init
from torch.nn.modules.batchnorm import _BatchNorm

class backbone(nn.Module):
    def __init__(self,
                 action_number_choosing,
                 rgb_detach=False,
                 pose_detach=False,
                 **args):
        super(backbone, self).__init__()
        print('Using I3D_TSA_fusion_concat backbone')

        self.rgb_detach = rgb_detach
        self.pose_detach = pose_detach

        if self.rgb_detach:
            print('rgb of pose_path\'s fusion block has been detached!')
        if self.pose_detach:
            print('pose of rgb_path\'s fusion block has been detached!')
        self.version = 'v2'
        self.fusion_block = args['fusion_block']
        
        self.rgb_path = I3D(**args['rgb_path'])
        self.pose_path = I3D(**args['pose_path'])
        Fusion_block = import_class('fusion_blocks.' + self.fusion_block)

        # self.rgb_fusion_layer1 = Fusion_block(in_channels = 64)
        self.rgb_fusion_layer2 = Fusion_block(in_channels = 192)
        self.rgb_fusion_layer3 = Fusion_block(in_channels = 480)
        # self.rgb_fusion_layer4 = Fusion_block(in_channels = 832, sub_sample = False)

        # self.pose_fusion_layer1 = Fusion_block(in_channels = 64)
        self.pose_fusion_layer2 = Fusion_block(in_channels = 192)
        self.pose_fusion_layer3 = Fusion_block(in_channels = 480)
        # self.pose_fusion_layer4 = Fusion_block(in_channels = 832, sub_sample = False)

        self.init_weights()

    def init_weights(self):
        """Initiate the parameters eitsher from existing checkpoint or from
        scratch."""
        Fusion_block = import_class('fusion_blocks.' + self.fusion_block)
        for m in self.modules():
            if isinstance(m, Fusion_block):
                for sub_m in m.modules():
                    if isinstance(sub_m, nn.Conv3d):
                        kaiming_init(sub_m)
                    elif isinstance(sub_m, _BatchNorm): 
                        constant_init(sub_m, 1)

        
    def forward(self, rgb_video_1, rgb_video_2, pose_video_1, pose_video_2):
        total_rgb_video = torch.cat((rgb_video_1, rgb_video_2), 0)
        total_pose_video = torch.cat((pose_video_1, pose_video_2), 0)
        # print('rgb_input',total_rgb_video.shape)
        # print('pose_input', total_pose_video.shape)
        #---------------------------------------------------------
        rgb = self.rgb_path.conv3d_1a_7x7(total_rgb_video)
        # print('rgb:',rgb.shape)
        rgb = self.rgb_path.maxPool3d_2a_3x3(rgb)
        # print('rgb:', rgb.shape)
        pose = self.pose_path.conv3d_1a_7x7(total_pose_video)
        # print('pose:', pose.shape)
        pose = self.pose_path.maxPool3d_2a_3x3(pose)
        # print('pose:', pose.shape)
        # feat_pose = pose.detach() if self.pose_detach else pose
        # rgb_f1 = self.rgb_fusion_layer1(rgb, feat_pose)
        # feat_rgb = rgb.detach() if self.rgb_detach else rgb
        # pose_f1 = self.pose_fusion_layer1(pose, feat_rgb)


        rgb = self.rgb_path.conv3d_2b_1x1(rgb)
        rgb = self.rgb_path.conv3d_2c_3x3(rgb)
        rgb = self.rgb_path.maxPool3d_3a_3x3(rgb)
        pose = self.pose_path.conv3d_2b_1x1(pose)
        pose = self.pose_path.conv3d_2c_3x3(pose)
        pose = self.pose_path.maxPool3d_3a_3x3(pose)

        feat_pose = pose.detach() if self.pose_detach else pose
        rgb_f2 = self.rgb_fusion_layer2(rgb, feat_pose)
        feat_rgb = rgb.detach() if self.rgb_detach else rgb
        pose_f2 = self.pose_fusion_layer2(pose, feat_rgb)

        rgb = self.rgb_path.mixed_3b(rgb_f2)
        rgb = self.rgb_path.mixed_3c(rgb)
        rgb = self.rgb_path.maxPool3d_4a_3x3(rgb)
        pose = self.pose_path.mixed_3b(pose_f2)
        pose = self.pose_path.mixed_3c(pose)
        pose = self.pose_path.maxPool3d_4a_3x3(pose)

        feat_pose = pose.detach() if self.pose_detach else pose
        rgb_f3 = self.rgb_fusion_layer3(rgb, feat_pose)
        feat_rgb = rgb.detach() if self.rgb_detach else rgb
        pose_f3 = self.pose_fusion_layer3(pose, feat_rgb)

        rgb = self.rgb_path.mixed_4b(rgb_f3)
        rgb = self.rgb_path.mixed_4c(rgb)
        rgb = self.rgb_path.mixed_4d(rgb)
        rgb = self.rgb_path.mixed_4e(rgb)
        rgb = self.rgb_path.mixed_4f(rgb)
        rgb = self.rgb_path.maxPool3d_5a_2x2(rgb)
        pose = self.pose_path.mixed_4b(pose_f3)
        pose = self.pose_path.mixed_4c(pose)
        pose = self.pose_path.mixed_4d(pose)
        pose = self.pose_path.mixed_4e(pose)
        pose = self.pose_path.mixed_4f(pose)
        pose = self.pose_path.maxPool3d_5a_2x2(pose)

        # feat_pose = pose.detach() if self.pose_detach else pose
        # rgb_f4 = self.rgb_fusion_layer4(rgb, feat_pose)
        # feat_rgb = rgb.detach() if self.rgb_detach else rgb
        # pose_f4 = self.pose_fusion_layer4(pose, feat_rgb)


        rgb = self.rgb_path.mixed_5b(rgb)
        feature_map_rgb = self.rgb_path.mixed_5c(rgb)
        feature_rgb = self.rgb_path.avg_pool(feature_map_rgb)
        
        pose = self.pose_path.mixed_5b(pose)
        feature_map_pose = self.pose_path.mixed_5c(pose)
        feature_pose = self.pose_path.avg_pool(feature_map_pose)


        #---------------------------------------------------------
        feature_rgb = feature_rgb.reshape(len(total_rgb_video), 1, -1) 
        Nt, C, T, H, W = feature_map_rgb.size()
        feature_map_rgb = feature_map_rgb.reshape(len(total_rgb_video), 1, C, T, H, W) 

        feature_1_rgb = feature_rgb[:feature_rgb.shape[0] // 2]
        feature_2_rgb = feature_rgb[feature_rgb.shape[0] // 2:]
        feamap_1_rgb = feature_map_rgb[:feature_map_rgb.shape[0] // 2]
        feamap_2_rgb = feature_map_rgb[feature_map_rgb.shape[0] // 2:]

        rgb_out = [feature_1_rgb, feature_2_rgb, feamap_1_rgb, feamap_2_rgb]

        feature_pose = feature_pose.reshape(len(total_pose_video), 1, -1) 
        Nt, C, T, H, W = feature_map_pose.size()
        feature_map_pose = feature_map_pose.reshape(len(total_pose_video), 1, C, T, H, W) 

        feature_1_pose = feature_pose[:feature_pose.shape[0] // 2]
        feature_2_pose = feature_pose[feature_pose.shape[0] // 2:]
        feamap_1_pose = feature_map_pose[:feature_map_pose.shape[0] // 2]
        feamap_2_pose = feature_map_pose[feature_map_pose.shape[0] // 2:]

        pose_out = [feature_1_pose, feature_2_pose, feamap_1_pose, feamap_2_pose]

        return rgb_out, pose_out


if __name__ == '__main__':
    import torch

    imgs_data = torch.randn(16, 3, 96, 56, 56)
    imgs_target = torch.randn(16, 3, 96, 56, 56)
    heatmap_imgs_data = torch.randn(16, 17, 96, 56, 56)
    heatmap_imgs_target = torch.randn(16, 17, 96, 56, 56)
    
    backbone_args = dict(
        rgb_path = dict(
            pretrained = '/data/guanjh/practice/pretrain/I3D/model_rgb.pth',
            in_channels=3),
        pose_path = dict(
            # pretrained = '/data/guanjh/practice/pretrain/I3D/model_pose.pth',
            in_channels=17),
        fusion_block = 'gaussian',
        action_number_choosing=True
    )
    net = backbone(**backbone_args)
    rgb_out, pose_out = net(imgs_data, imgs_target, heatmap_imgs_data, heatmap_imgs_target)
    # for rgb in rgb_out:
    #     print(rgb.size())
    # for pose in pose_out:
    #     print(pose.size())