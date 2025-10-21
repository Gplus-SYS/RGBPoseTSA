model = dict(
    backbone=dict(
        type='baseline',
        # pretrained = 'pretrain/I3D/model_pose.pth',
        in_channels=17),
    cls_head=dict(
        type='I3DHead',
        in_channels=1024,
        num_classes=52,
        dropout=0.5),
    ps_net=dict(
        type='PSNet',
        frames=9,
        dropout=0),
    decoder=dict(
        type ='decoder_fuser',
        dim=64, 
        num_heads=8, 
        num_layers=3),
    regressor=dict(
        type='MLP_score',
        in_channel=64, 
        out_channel=1))

cls_only = False
ps = True
with_cls = False

dataset_type = 'PoseDataset'
# dataset
ann_file = '/data/FineDiving_RGBPose_annotation.pkl'
train_split_pkl = '/data/FineDiving/Annotations/train_split.pkl'
test_split_pkl = '/data/FineDiving/Annotations/test_split.pkl'

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
left_limb = [0, 2, 3, 6, 7, 8, 12, 14]
right_limb = [1, 4, 5, 9, 10, 11, 13, 15]
skeletons = [[0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11],
             [11, 13], [13, 15], [6, 12], [12, 14], [14, 16], [0, 1], [0, 2],
             [1, 3], [2, 4], [11, 12]]


action_number_choosing = True
voter_number = 10
fix_size = 5
step_num = 3
prob_tas_threshold = 0.25

train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=96, keyframes=0, regular = True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', shape=(64, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', shape=(56, 56)),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GenerateHeatmaps', with_kp=True, with_limb=False),
    # dict(type='GenerateHeatmaps', with_kp=False, with_limb=True, skeletons=skeletons),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label', 'transits', 'dive_score'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'transits', 'dive_score'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=96, keyframes=0, regular = True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', shape=(56, 56)),
    dict(type='GenerateHeatmaps', with_kp=True, with_limb=False),
    # dict(type='GenerateHeatmaps', with_kp=False, with_limb=True, skeletons=skeletons),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label', 'transits', 'dive_score'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'transits', 'dive_score'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=96, keyframes=0, regular = True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', shape=(56, 56)),
    dict(type='GenerateHeatmaps', with_kp=True, with_limb=False),
    # dict(type='GenerateHeatmaps', with_kp=False, with_limb=True, skeletons=skeletons,
    #      double=True, left_kp=left_kp, right_kp=right_kp, left_limb=left_limb, right_limb=right_limb),
    dict(type='FormatShape', input_format='NCTHW_Heatmap'),
    dict(type='Collect', keys=['imgs', 'label', 'transits', 'dive_score'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'transits', 'dive_score'])
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1,
                         workers_per_gpu=2),
    train=dict(ann_file=ann_file, 
               train_split_pkl=train_split_pkl, 
               test_split_pkl=test_split_pkl, 
               split='train', 
               repeat=10),
    val=dict(ann_file=ann_file, 
             train_split_pkl=train_split_pkl,
             test_split_pkl=test_split_pkl, 
             split='test', 
             repeat=1),
    test=dict(ann_file=ann_file, 
              train_split_pkl=train_split_pkl, 
              test_split_pkl=test_split_pkl, 
              split='test', 
              repeat=1))

seed = 0
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0003)
# optimizer = dict(type='Adam', base_lr=0.001, lr_factor=0.1, weight_decay=0.0003)
# grad_clip=dict(max_norm=40, norm_type=2)

# learning policy
lr_config = dict(enable=True, policy='CosineAnnealing', by_epoch=False, warmup_steps=5, min_lr=0)
total_epochs = 24
fix_bn = False
print_freq = 100
work_dir = 'experiments/e1'
resume = False


## bash train.sh PoseI3D configs/FineDiving_PoseI3D.py 3
