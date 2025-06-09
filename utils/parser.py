import os
from mmcv import Config
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--archs', type = str, default='RGBI3D', choices=['TSA','PoseC3D','PoseI3D','RGBI3D','RGBPoseI3D_late_fusion', 'RGBPoseI3D_early_fusion'], help = 'our approach')
    parser.add_argument('--config', type=str, default='configs/FineDiving_RGBI3D.py',help = 'config of experiment')
    parser.add_argument('--prefix', type = str, default='default', help = 'experiment name')
    parser.add_argument('--resume', action='store_true', default=False ,help = 'resume training (interrupted by accident)')
    parser.add_argument('--sync_bn', type=bool, default=False)
    parser.add_argument('--fix_bn', type=bool, default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--ckpts', type=str, default=None, help='test used ckpt path')
    parser.add_argument('--timestamp', type=str, default=None, help='timestamp')
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    if args.test:
        if args.ckpts is None:
            raise RuntimeError('--ckpts should not be None when --test is activate')

    return args

def setup(args):

    if args.resume:
        cfg_path = os.path.join(args.experiment_path,'config.py')
        if not os.path.exists(cfg_path):
            print("Failed to resume")
            args.resume = False
            setup(args)
            return
        print('Resume yaml from %s' % cfg_path)
        config = Config.fromfile(cfg_path)
        merge_config(config, args)
        args.experiment_path = args.work_dir
        args.resume = True
    else:
        config = get_config(args)
        merge_config(config, args)
        args.experiment_path = args.work_dir
        create_experiment_dir(args)
        save_experiment_config(args,config)

def get_config(args):
    try:
        print('Load config from %s' % args.config)
        config = Config.fromfile(args.config)
    except:
        raise NotImplementedError('%s arch is not supported'% args.archs)
    return config

def merge_config(config, args):
    for k, v in config.items():
        setattr(args, k, v)   

def create_experiment_dir(args):
    try:
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    except:
        pass
    
def save_experiment_config(args,config):
    config_path = os.path.join(args.experiment_path,'config.py')
    config.dump(config_path)
    print('Save the Config file at %s' % config_path)