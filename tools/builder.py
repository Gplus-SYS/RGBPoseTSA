import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))

import copy
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from utils.misc import import_class


def pipelines_builder(args):
    train_pipelines, val_pipelines, test_pipelines = [], [], []

    for arg in args.train_pipeline:
        arg = copy.deepcopy(arg)
        Pipe_Line = import_class('datasets.pipelines.' + arg.get('type'))
        del arg['type']
        pipe_line = Pipe_Line(**arg)
        train_pipelines.append(pipe_line)

    for arg in args.val_pipeline:
        arg = copy.deepcopy(arg)
        Pipe_Line = import_class('datasets.pipelines.' + arg.get('type'))
        del arg['type']
        pipe_line = Pipe_Line(**arg)
        val_pipelines.append(pipe_line)

    for arg in args.test_pipeline:
        arg = copy.deepcopy(arg)
        Pipe_Line = import_class('datasets.pipelines.' + arg.get('type'))
        del arg['type']
        pipe_line = Pipe_Line(**arg)
        test_pipelines.append(pipe_line)

    return transforms.Compose(train_pipelines), transforms.Compose(val_pipelines), transforms.Compose(test_pipelines)

def dataset_builder(args):
    train_pipelines, val_pipelines, test_pipelines = pipelines_builder(args)
    Dataset = import_class('datasets.' + args.dataset_type)
    train_dataset = Dataset(args.seed, args.action_number_choosing, args.voter_number, args.data.train, train_pipelines)
    val_dataset = Dataset(args.seed, args.action_number_choosing, args.voter_number, args.data.val, val_pipelines)
    test_dataset = Dataset(args.seed, args.action_number_choosing, args.voter_number, args.data.test, test_pipelines)
    return train_dataset, val_dataset, test_dataset

def model_builder(args):
    args = copy.deepcopy(args)
    multi_modality = args.model.get('multi_modality', False)
    action_number_choosing = args.action_number_choosing
    if multi_modality :
        if multi_modality == 'late fusion':
            rgb_Backbone = import_class('models.' + args.model.backbone.rgb_path.type)
            del args.model.backbone.rgb_path.type
            rgb_backbone = rgb_Backbone(action_number_choosing, **args.model.backbone.rgb_path)
            print('I3D\'s version is',rgb_backbone.version)
            
            pose_Backbone = import_class('models.' + args.model.backbone.pose_path.type)
            del args.model.backbone.pose_path.type
            pose_backbone = pose_Backbone(action_number_choosing, **args.model.backbone.pose_path)

        else:
            early_fuision_Backbone = import_class('models.' + multi_modality)
            backbone = early_fuision_Backbone(action_number_choosing, **args.model.backbone)
            print('I3D\'s version is',backbone.version)
            print('fusion block is',backbone.fusion_block)
        

        if (args.cls_only):
            Head = import_class('models.' + args.model.cls_head.type)
            del args.model.cls_head.type
            cls_head = Head(**args.model.cls_head)
            if multi_modality == 'late fusion':
                return rgb_backbone, pose_backbone, cls_head
            else:
                return backbone, cls_head
        
        PSNet = import_class('models.' + args.model.ps_net.type)
        del args.model.ps_net.type
        ps_net = PSNet(**args.model.ps_net)

        Decoder = import_class('models.' + args.model.decoder.type)
        del args.model.decoder.type
        decoder = Decoder(**args.model.decoder)

        Regressor = import_class('models.' + args.model.regressor.type)
        del args.model.regressor.type
        regressor = Regressor(**args.model.regressor)
        
        if multi_modality == 'late fusion':
            return rgb_backbone, pose_backbone, ps_net, decoder, regressor
        else:
            updata_path = args.model.updata_by_which_path
            assert updata_path in ['RGB','Pose','Double']
            if updata_path in ['RGB','Pose']:
                print('All parameters are updated by %s\'s path loss!'%(updata_path))
                return backbone, ps_net, decoder, regressor
            else:
                print('The parameters on the RGB and Pose paths are updated using their respective losses!')
                ps_net_2 = PSNet(**args.model.ps_net)
                decoder_2 = Decoder(**args.model.decoder)
                regressor_2 = Regressor(**args.model.regressor)
                return backbone, ps_net, ps_net_2, decoder, decoder_2, regressor, regressor_2

    else:
        Backbone = import_class('models.' + args.model.backbone.type)
        del args.model.backbone.type
        backbone = Backbone(action_number_choosing, **args.model.backbone)
        print('I3D\'s version is',backbone.version)

        if (args.cls_only):
            Head = import_class('models.' + args.model.cls_head.type)
            del args.model.cls_head.type
            cls_head = Head(**args.model.cls_head)
            return backbone, cls_head

        PSNet = import_class('models.' + args.model.ps_net.type)
        del args.model.ps_net.type
        
        ps_net = PSNet(**args.model.ps_net)
        if (args.ps):
            return backbone, ps_net

        Decoder = import_class('models.' + args.model.decoder.type)
        del args.model.decoder.type
        decoder = Decoder(**args.model.decoder)

        Regressor = import_class('models.' + args.model.regressor.type)
        del args.model.regressor.type
        regressor = Regressor(**args.model.regressor)

        if args.with_cls:
            Head = import_class('models.' + args.model.cls_head.type)
            del args.model.cls_head.type
            cls_head = Head(**args.model.cls_head)
            return backbone, cls_head, ps_net, decoder, regressor

        return backbone, ps_net, decoder, regressor

def optimizer_scheduler_builder(models, args):
    optimizer_cfg = args.optimizer
    optimizer_type = optimizer_cfg.get('type')
    optimizer = None
    scheduler = None
    model_params = []
    
    for model in models:
        model_param = {'params': model.parameters()}
        model_params.append(model_param)
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model_params, 
                              lr=optimizer_cfg.get('lr'), 
                              momentum=optimizer_cfg.get('momentum'), 
                              weight_decay=optimizer_cfg.get('weight_decay'))
    elif optimizer_type == 'Adam':
        model_params[0]['lr'] = optimizer_cfg.base_lr * optimizer_cfg.lr_factor
        multi_modality = args.model.get('multi_modality', False)
        if multi_modality:
            model_params[1]['lr'] = optimizer_cfg.base_lr * optimizer_cfg.lr_factor
        optimizer = optim.Adam(model_params,
                               lr=optimizer_cfg.base_lr, 
                               weight_decay=optimizer_cfg.weight_decay)

    scheduler_cfg = args.lr_config
    scheduler_policy = scheduler_cfg.get('policy')
    scheduler = None
    if scheduler_policy == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optimizer=optimizer,
                                    T_max=args.total_epochs,
                                    eta_min=scheduler_cfg.get('min_lr'))
    return optimizer, scheduler



def load_model(backbone, ps_net, decoder, regressor, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    backbone_ckpt = {k.replace("module.", ""): v for k, v in state_dict['backbone'].items()}
    backbone.load_state_dict(backbone_ckpt)

    ps_net_ckpt = {k.replace("module.", ""): v for k, v in state_dict['ps_net'].items()}
    ps_net.load_state_dict(ps_net_ckpt)

    decoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['decoder'].items()}
    decoder.load_state_dict(decoder_ckpt)

    regressor_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor'].items()}
    regressor.load_state_dict(regressor_ckpt)    

    print('Load Success!')

def load_model_c(backbone, cls_head, ps_net, decoder, regressor, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    backbone_ckpt = {k.replace("module.", ""): v for k, v in state_dict['backbone'].items()}
    backbone.load_state_dict(backbone_ckpt)

    cls_head_ckpt = {k.replace("module.", ""): v for k, v in state_dict['cls_head'].items()}
    cls_head.load_state_dict(cls_head_ckpt)

    ps_net_ckpt = {k.replace("module.", ""): v for k, v in state_dict['ps_net'].items()}
    ps_net.load_state_dict(ps_net_ckpt)

    decoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['decoder'].items()}
    decoder.load_state_dict(decoder_ckpt)

    regressor_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor'].items()}
    regressor.load_state_dict(regressor_ckpt)

    print('Load Success!')

def load_model_double_path(backbone, ps_net_rgb, ps_net_pose, decoder_rgb, decoder_pose, regressor_rgb, regressor_pose, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    backbone_ckpt = {k.replace("module.", ""): v for k, v in state_dict['backbone'].items()}
    backbone.load_state_dict(backbone_ckpt)

    ps_net_rgb_ckpt = {k.replace("module.", ""): v for k, v in state_dict['ps_net_rgb'].items()}
    ps_net_rgb.load_state_dict(ps_net_rgb_ckpt)
    ps_net_pose_ckpt = {k.replace("module.", ""): v for k, v in state_dict['ps_net_pose'].items()}
    ps_net_pose.load_state_dict(ps_net_pose_ckpt)

    decoder_rgb_ckpt = {k.replace("module.", ""): v for k, v in state_dict['decoder_rgb'].items()}
    decoder_rgb.load_state_dict(decoder_rgb_ckpt)
    decoder_pose_ckpt = {k.replace("module.", ""): v for k, v in state_dict['decoder_pose'].items()}
    decoder_pose.load_state_dict(decoder_pose_ckpt)

    regressor_rgb_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor_rgb'].items()}
    regressor_rgb.load_state_dict(regressor_rgb_ckpt)
    regressor_pose_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor_pose'].items()}
    regressor_pose.load_state_dict(regressor_pose_ckpt)

    print('Load Success!')

def load_cls_model(backbone, cls_head, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    backbone_ckpt = {k.replace("module.", ""): v for k, v in state_dict['backbone'].items()}
    backbone.load_state_dict(backbone_ckpt)

    cls_head_ckpt = {k.replace("module.", ""): v for k, v in state_dict['cls_head'].items()}
    cls_head.load_state_dict(cls_head_ckpt)

    
    print('Load Success!')


def load_ps_model(backbone, ps_net, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    backbone_ckpt = {k.replace("module.", ""): v for k, v in state_dict['backbone'].items()}
    backbone.load_state_dict(backbone_ckpt)

    ps_net_ckpt = {k.replace("module.", ""): v for k, v in state_dict['ps_net'].items()}
    ps_net.load_state_dict(ps_net_ckpt)

    print('Load Success!')


def MM_load_model(rgb_backbone, pose_backbone, ps_net, decoder, regressor, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    rgb_backbone_ckpt = {k.replace("module.", ""): v for k, v in state_dict['rgb_backbone'].items()}
    rgb_backbone.load_state_dict(rgb_backbone_ckpt)

    pose_backbone_ckpt = {k.replace("module.", ""): v for k, v in state_dict['pose_backbone'].items()}
    pose_backbone.load_state_dict(pose_backbone_ckpt)

    ps_net_ckpt = {k.replace("module.", ""): v for k, v in state_dict['ps_net'].items()}
    ps_net.load_state_dict(ps_net_ckpt)

    decoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['decoder'].items()}
    decoder.load_state_dict(decoder_ckpt)

    regressor_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor'].items()}
    regressor.load_state_dict(regressor_ckpt)
    
    print('Load Success!')

def MM_load_cls_model(rgb_backbone, pose_backbone, cls_head, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print('Loading weights from %s...' % ckpt_path)

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    rgb_backbone_ckpt = {k.replace("module.", ""): v for k, v in state_dict['rgb_backbone'].items()}
    rgb_backbone.load_state_dict(rgb_backbone_ckpt)

    pose_backbone_ckpt = {k.replace("module.", ""): v for k, v in state_dict['pose_backbone'].items()}
    pose_backbone.load_state_dict(pose_backbone_ckpt)

    cls_head_ckpt = {k.replace("module.", ""): v for k, v in state_dict['cls_head'].items()}
    cls_head.load_state_dict(cls_head_ckpt)

    
    print('Load Success!')

def resume_train(backbone, ps_net, decoder, regressor, optimizer, scheduler, args):
    ckpt_path = os.path.join(args.experiment_path, 'last.pth')
    assert os.path.exists(ckpt_path), 'no checkpoint file from path %s...' % ckpt_path
    print('Loading weights from %s...' % ckpt_path)

     # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    backbone_ckpt = {k.replace("module.", ""): v for k, v in state_dict['backbone'].items()}
    backbone.load_state_dict(backbone_ckpt)

    ps_net_ckpt = {k.replace("module.", ""): v for k, v in state_dict['ps_net'].items()}
    ps_net.load_state_dict(ps_net_ckpt)

    decoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['decoder'].items()}
    decoder.load_state_dict(decoder_ckpt)

    regressor_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor'].items()}
    regressor.load_state_dict(regressor_ckpt)

    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

    scheduler.load_state_dict(state_dict['scheduler'])

    start_epoch = state_dict['epoch'] + 1
    epoch_best_aqa = state_dict['epoch_best_aqa']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']

    return start_epoch, epoch_best_aqa, rho_best, L2_min, RL2_min

def resume_train_double_path(backbone, ps_net_rgb, ps_net_pose, decoder_rgb, decoder_pose, 
                             regressor_rgb, regressor_pose, optimizer, scheduler, args):
    ckpt_path = os.path.join(args.experiment_path, 'last.pth')
    assert os.path.exists(ckpt_path), 'no checkpoint file from path %s...' % ckpt_path
    print('Loading weights from %s...' % ckpt_path)

     # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    backbone_ckpt = {k.replace("module.", ""): v for k, v in state_dict['backbone'].items()}
    backbone.load_state_dict(backbone_ckpt)

    ps_net_rgb_ckpt = {k.replace("module.", ""): v for k, v in state_dict['ps_net_rgb'].items()}
    ps_net_rgb.load_state_dict(ps_net_rgb_ckpt)
    ps_net_pose_ckpt = {k.replace("module.", ""): v for k, v in state_dict['ps_net_pose'].items()}
    ps_net_pose.load_state_dict(ps_net_pose_ckpt)

    decoder_rgb_ckpt = {k.replace("module.", ""): v for k, v in state_dict['decoder_rgb'].items()}
    decoder_rgb.load_state_dict(decoder_rgb_ckpt)
    decoder_pose_ckpt = {k.replace("module.", ""): v for k, v in state_dict['decoder_pose'].items()}
    decoder_pose.load_state_dict(decoder_pose_ckpt)

    regressor_rgb_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor_rgb'].items()}
    regressor_rgb.load_state_dict(regressor_rgb_ckpt)
    regressor_pose_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor_pose'].items()}
    regressor_pose.load_state_dict(regressor_pose_ckpt)
    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

    scheduler.load_state_dict(state_dict['scheduler'])

    start_epoch = state_dict['epoch'] + 1
    epoch_best_aqa_rgb = state_dict['epoch_best_aqa_rgb']
    epoch_best_aqa_pose = state_dict['epoch_best_aqa_pose']
    rho_best_rgb = state_dict['rho_best_rgb']
    rho_best_pose = state_dict['rho_best_pose']
    L2_min_rgb = state_dict['L2_min_rgb']
    L2_min_pose = state_dict['L2_min_pose']
    RL2_min_rgb = state_dict['RL2_min_rgb']
    RL2_min_pose = state_dict['RL2_min_pose']
    return start_epoch, epoch_best_aqa_rgb, epoch_best_aqa_pose, rho_best_rgb, rho_best_pose, L2_min_rgb, L2_min_pose, RL2_min_rgb, RL2_min_pose

def MM_resume_train(rgb_backbone, pose_backbone, ps_net, decoder, regressor, optimizer, scheduler, args):
    ckpt_path = os.path.join(args.experiment_path, 'last.pth')
    if not os.path.exists(ckpt_path):
        print('no checkpoint file from path %s...' % ckpt_path)
        return 0, 0
    print('Loading weights from %s...' % ckpt_path)

     # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    rgb_backbone_ckpt = {k.replace("module.", ""): v for k, v in state_dict['rgb_backbone'].items()}
    rgb_backbone.load_state_dict(rgb_backbone_ckpt)

    pose_backbone_ckpt = {k.replace("module.", ""): v for k, v in state_dict['pose_backbone'].items()}
    pose_backbone.load_state_dict(pose_backbone_ckpt)

    ps_net_ckpt = {k.replace("module.", ""): v for k, v in state_dict['ps_net'].items()}
    ps_net.load_state_dict(ps_net_ckpt)

    decoder_ckpt = {k.replace("module.", ""): v for k, v in state_dict['decoder'].items()}
    decoder.load_state_dict(decoder_ckpt)

    regressor_ckpt = {k.replace("module.", ""): v for k, v in state_dict['regressor'].items()}
    regressor.load_state_dict(regressor_ckpt)

    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

    scheduler.load_state_dict(state_dict['scheduler'])

    start_epoch = state_dict['epoch'] + 1
    epoch_best_aqa = state_dict['epoch_best_aqa']
    rho_best = state_dict['rho_best']
    L2_min = state_dict['L2_min']
    RL2_min = state_dict['RL2_min']

    return start_epoch, epoch_best_aqa, rho_best, L2_min, RL2_min

def resume_cls_train(backbone, cls_head, optimizer, scheduler, args):
    ckpt_path = os.path.join(args.experiment_path, 'last.pth')
    if not os.path.exists(ckpt_path):
        print('no checkpoint file from path %s...' % ckpt_path)
        return 0, 0
    print('Loading weights from %s...' % ckpt_path)

     # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    backbone_ckpt = {k.replace("module.", ""): v for k, v in state_dict['backbone'].items()}
    backbone.load_state_dict(backbone_ckpt)

    cls_head_ckpt = {k.replace("module.", ""): v for k, v in state_dict['cls_head'].items()}
    cls_head.load_state_dict(cls_head_ckpt)

    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

    scheduler.load_state_dict(state_dict['scheduler'])

    start_epoch = state_dict['epoch'] + 1
    best_top1_acc = state_dict['best_top1_acc']

    return start_epoch, best_top1_acc


def MM_resume_cls_train(rgb_backbone, pose_backbone, cls_head, optimizer, scheduler, args):
    ckpt_path = os.path.join(args.experiment_path, 'last.pth')
    if not os.path.exists(ckpt_path):
        print('no checkpoint file from path %s...' % ckpt_path)
        return 0, 0
    print('Loading weights from %s...' % ckpt_path)

     # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    rgb_backbone_ckpt = {k.replace("module.", ""): v for k, v in state_dict['rgb_backbone'].items()}
    rgb_backbone.load_state_dict(rgb_backbone_ckpt)

    pose_backbone_ckpt = {k.replace("module.", ""): v for k, v in state_dict['pose_backbone'].items()}
    pose_backbone.load_state_dict(pose_backbone_ckpt)

    cls_head_ckpt = {k.replace("module.", ""): v for k, v in state_dict['cls_head'].items()}
    cls_head.load_state_dict(cls_head_ckpt)

    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

    scheduler.load_state_dict(state_dict['scheduler'])

    start_epoch = state_dict['epoch'] + 1
    best_top1_acc = state_dict['best_top1_acc']

    return start_epoch, best_top1_acc

def resume_ps_train(backbone, ps_net, optimizer, scheduler, args):
    ckpt_path = os.path.join(args.experiment_path, 'last.pth')
    if not os.path.exists(ckpt_path):
        print('no checkpoint file from path %s...' % ckpt_path)
        return 0, 0
    print('Loading weights from %s...' % ckpt_path)

     # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')

    backbone_ckpt = {k.replace("module.", ""): v for k, v in state_dict['backbone'].items()}
    backbone.load_state_dict(backbone_ckpt)

    ps_net_ckpt = {k.replace("module.", ""): v for k, v in state_dict['ps_net'].items()}
    ps_net.load_state_dict(ps_net_ckpt)

    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

    scheduler.load_state_dict(state_dict['scheduler'])

    start_epoch = state_dict['epoch'] + 1
    pred_tious_best_5 = state_dict['pred_tious_best_5']
    pred_tious_best_75 = state_dict['pred_tious_best_75']

    return start_epoch, pred_tious_best_5, pred_tious_best_75


    
    
