model = dict(
    backbone=dict(
        type='baseline',
        pretrained = 'pretrain/I3D/model_rgb.pth',
        in_channels=3),
    cls_head=dict(
        type='I3DHead',
        in_channels=1024,
        num_classes=52,
        dropout=0.5),
    ps_net=dict(
        type='PSNet',
        frames = 9,
        dropout = 0),
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
ps = False
with_cls = True

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

dataset_type = 'RGBDataset'
# dataset
data_root = '/data/FineDiving/datasets/FinaDiving/Trimmed_Video_Frames/FINADiving_MTL_256s'
ann_file = '/data/FineDiving_RGBPose_annotation.pkl'
train_split_pkl = '/data/FineDiving/Annotations/train_split.pkl'
test_split_pkl = '/data/FineDiving/Annotations/test_split.pkl'


action_number_choosing = True
voter_number = 10
fix_size = 5
step_num = 3
prob_tas_threshold = 0.25

train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=96, keyframes=0, regular = True),
    dict(type='RGBDecode'),
    dict(type='Resize', shape=(64, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', shape=(56, 56)),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label', 'transits', 'dive_score'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'transits', 'dive_score'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=96, keyframes=0, regular = True),
    dict(type='RGBDecode'),
    dict(type='Resize', shape=(56, 56)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label', 'transits', 'dive_score'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'transits', 'dive_score'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=96, keyframes=0, regular = True),
    dict(type='RGBDecode'),
    dict(type='Resize', shape=(56, 56)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label', 'transits', 'dive_score'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label', 'transits', 'dive_score'])
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1,
                         workers_per_gpu=2),
    train=dict(ann_file=ann_file, 
               data_root=data_root, 
               train_split_pkl=train_split_pkl, 
               test_split_pkl=test_split_pkl, 
               split='train', 
               repeat=10),
    val=dict(ann_file=ann_file, 
             data_root=data_root, 
             train_split_pkl=train_split_pkl,
             test_split_pkl=test_split_pkl, 
             split='test', 
             repeat=1),
    test=dict(ann_file=ann_file, 
              data_root=data_root, 
              train_split_pkl=train_split_pkl, 
              test_split_pkl=test_split_pkl, 
              split='test', 
              repeat=1))

seed = 0
# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0003)
optimizer = dict(type='Adam', base_lr=0.001, lr_factor=0.1, weight_decay=0.0003)
# grad_clip=dict(max_norm=40, norm_type=2)

# learning policy
lr_config = dict(enable=True, policy='CosineAnnealing', by_epoch=False, min_lr=0)
total_epochs = 24
fix_bn = True
print_freq = 100
work_dir = 'experiments/e1'

resume = False

## bash train.sh RGBI3D configs/FineDiving_RGBI3D.py 1





