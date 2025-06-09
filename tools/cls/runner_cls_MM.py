import numpy as np
import random
import os
import torch
import torch.nn as nn
from datetime import datetime
from tools import builder
import helper_cls_MM as helper
from utils import misc
from scipy import stats
from torch.cuda.amp import GradScaler
import time

def train_net(args):
    global action_number_choosing, use_gpu, best_top1_acc
    action_number_choosing = args.action_number_choosing
    use_gpu = torch.cuda.is_available()
    best_top1_acc = 0
    print('Trainer start ... ')
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # build dataset
    train_dataset, val_dataset, _ = builder.dataset_builder(args)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.data.videos_per_gpu,
                                            shuffle=True, num_workers=int(args.data.workers_per_gpu),
                                            pin_memory=True, worker_init_fn=misc.worker_init_fn)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.data.test_dataloader.videos_per_gpu,
                                                    shuffle=False, num_workers=int(args.data.test_dataloader.workers_per_gpu),
                                                    pin_memory=True)
    # build model
    rgb_backbone, pose_backbone, cls_head = builder.model_builder(args)
    # build opti
    optimizer, scheduler = builder.optimizer_scheduler_builder([rgb_backbone, pose_backbone, cls_head], args)
    if use_gpu:
        rgb_backbone.cuda()
        pose_backbone.cuda()
        cls_head.cuda()
    # DP
    rgb_backbone = nn.DataParallel(rgb_backbone)
    pose_backbone = nn.DataParallel(pose_backbone)
    cls_head = nn.DataParallel(cls_head)
    scaler = GradScaler()
    # resume    
    start_epoch = 0
    if args.resume:
        start_epoch, best_top1_acc = builder.MM_resume_cls_train(rgb_backbone, pose_backbone, cls_head, optimizer, scheduler, args)
            
    for epoch in range(start_epoch, args.total_epochs):
        epoch_start_time = time.time()
        top1_accs = []
        rgb_backbone.train()
        pose_backbone.train()
        cls_head.train()
        if args.fix_bn:
            rgb_backbone.apply(misc.fix_bn)
            pose_backbone.apply(misc.fix_bn)
        # training epoch
        if action_number_choosing:
            for idx, (data, target) in enumerate(train_dataloader):
                helper.MM_backbone_forward_train(rgb_backbone, pose_backbone, cls_head, data, target, optimizer, epoch, 
                                                    idx+1, len(train_dataloader), scaler, top1_accs, args)
        else:
            for idx, data in enumerate(train_dataloader):
                helper.MM_backbone_forward_train_single(rgb_backbone, pose_backbone, cls_head, data, optimizer, epoch, 
                                                    idx+1, len(train_dataloader), scaler, top1_accs, args)
        # evaluation results
        top1_acc = sum(top1_accs) / len(train_dataloader) 
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time       
        print('[Time: %s][Training] EPOCH: %d, epoch time: %dmin%ds ,top1_acc: %.4f'
              % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, epoch_time//60, epoch_time%60, top1_acc))
        # validate
        validate_net(rgb_backbone, pose_backbone, cls_head, val_dataloader, epoch, optimizer, scheduler, args)
        helper.MM_save_cls_checkpoint(rgb_backbone, pose_backbone, cls_head, optimizer, scheduler, epoch, top1_acc, 'last', args)
        # scheduler lr
        scheduler_cfg = args.lr_config
        scheduler_enable = scheduler_cfg.get('enable')
        if scheduler_enable:
            scheduler.step()
    # Test
    print('[Time: %s][Testing last]' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    test_net(args, os.path.join(args.experiment_path, 'last.pth'))
    print('[Time: %s][Testing best]' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    test_net(args, os.path.join(args.experiment_path, 'best.pth'))

            
def validate_net(rgb_backbone, pose_backbone, cls_head, val_dataloader, epoch, optimizer, scheduler, args):
    print("Start validating epoch {} ......".format(epoch))
    global use_gpu, best_top1_acc
    top1_accs = []
    rgb_backbone.eval()
    pose_backbone.eval()
    cls_head.eval()
    with torch.no_grad():
        preds = []
        gt_labels = []
        validate_start_time = time.time()
        if action_number_choosing:
            for _, (data, targets) in enumerate(val_dataloader, 0):
                pred, gt_label = helper.MM_backbone_forward_test(rgb_backbone, pose_backbone, cls_head, data, targets, top1_accs)
                preds.extend(pred)
                gt_labels.extend(gt_label)
        else:
            for _, data in enumerate(val_dataloader, 0):
                pred, gt_label = helper.MM_backbone_forward_test_single(rgb_backbone, pose_backbone, cls_head, data, top1_accs)
                preds.extend(pred)
                gt_labels.extend(gt_label)
        # evaluation results
        validate_end_time = time.time()
        top1_acc = sum(top1_accs) / len(val_dataloader)
        validate_time = validate_end_time - validate_start_time
        print('[Time: %s][Validating] EPOCH: %d, validate time: %dmin%ds, top1_acc: %.4f' % 
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, validate_time//60, validate_time%60, top1_acc))
        if top1_acc > best_top1_acc:
            best_top1_acc = top1_acc
            print('-----New best found!-----')
            helper.save_cls_outputs(preds,gt_labels,args)
            helper.MM_save_cls_checkpoint(rgb_backbone, pose_backbone, cls_head, optimizer, scheduler, best_top1_acc, epoch, 'best', args)
            

def test_net(args, ckpt_path = None):
    print('Tester start ...... ')
    global action_number_choosing
    action_number_choosing = args.action_number_choosing
    # CUDA
    global use_gpu
    use_gpu = torch.cuda.is_available()
    # build dataset
    _, _, test_dataset = builder.dataset_builder(args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.data.test_dataloader.videos_per_gpu,
                                                    shuffle=False, num_workers=int(args.data.test_dataloader.workers_per_gpu),
                                                    pin_memory=True)
    # build model
    rgb_backbone, pose_backbone, cls_head = builder.model_builder(args)
    if ckpt_path:
        builder.MM_load_cls_model(rgb_backbone, pose_backbone, cls_head, ckpt_path)
    else:
        builder.MM_load_cls_model(rgb_backbone, pose_backbone, cls_head, args.ckpts)
    if use_gpu:
        rgb_backbone.cuda()
        pose_backbone.cuda()
        cls_head.cuda()
        torch.backends.cudnn.benchmark = True
    # DP
    rgb_backbone = nn.DataParallel(rgb_backbone)
    pose_backbone = nn.DataParallel(pose_backbone)
    cls_head = nn.DataParallel(cls_head)
    test(rgb_backbone, pose_backbone, cls_head, test_dataloader, args)

def test(rgb_backbone, pose_backbone, cls_head, test_dataloader, args):
    global use_gpu
    top1_accs = []
    rgb_backbone.eval()
    pose_backbone.eval()
    cls_head.eval()
    with torch.no_grad():
        test_start_time = time.time()
        if action_number_choosing:
            for _, (data, targets) in enumerate(test_dataloader, 0):
                _, _ = helper.MM_backbone_forward_test(rgb_backbone, pose_backbone, cls_head, 
                                                            data, targets, top1_accs)
        else:
            for _, data in enumerate(test_dataloader, 0):
                _, _ = helper.MM_backbone_forward_test_single(rgb_backbone, pose_backbone, cls_head, 
                                                            data, top1_accs)
    test_end_time = time.time()
    top1_acc = sum(top1_accs) / len(test_dataloader)
    test_time = test_end_time - test_start_time
    print('[Time: %s][TEST] test time: %dmin%ds, top1_acc: %.4f' % 
          (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), test_time//60, test_time%60, top1_acc))
