import numpy as np
import random
import os
import torch
import torch.nn as nn
from datetime import datetime
from tools import builder
from tools.cls import helper_cls as helper
from utils import misc
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
    backbone, cls_head = builder.model_builder(args)
    # build opti
    optimizer, scheduler = builder.optimizer_scheduler_builder([backbone, cls_head],args)
    if use_gpu:
        backbone.cuda()
        cls_head.cuda()
    # DP
    backbone = nn.DataParallel(backbone)
    cls_head = nn.DataParallel(cls_head)
    scaler = GradScaler()
    # resume    
    start_epoch = 0
    if args.resume:
        start_epoch, best_top1_acc = builder.resume_cls_train(backbone, cls_head, optimizer, scheduler, args)
    for epoch in range(start_epoch, args.total_epochs):
        epoch_start_time = time.time()
        top1_accs = []
        backbone.train()
        cls_head.train()
        if args.fix_bn:
            backbone.apply(misc.fix_bn)
        # training epoch
        if action_number_choosing:
            for idx, (data, target) in enumerate(train_dataloader):
                helper.backbone_forward_train(backbone, cls_head, data, target, optimizer, epoch, 
                                                      idx+1, len(train_dataloader), scaler, top1_accs, args)
        else:
            for idx, data in enumerate(train_dataloader):
                helper.backbone_forward_train_single(backbone, cls_head, data, optimizer, epoch, 
                                                      idx+1, len(train_dataloader), scaler, top1_accs, args)
        # evaluation results
        top1_acc = sum(top1_accs) / len(train_dataloader)
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        print('[Time: %s][Training] EPOCH: %d, epoch time: %dmin%ds ,top1_acc: %.4f'
              % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, epoch_time//60, epoch_time%60, top1_acc))
        # validate
        validate_net(backbone, cls_head, val_dataloader, epoch, optimizer, scheduler, args)
        helper.save_cls_checkpoint(backbone, cls_head, optimizer, scheduler, epoch, top1_acc, 'last', args)
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

def validate_net(backbone, cls_head, val_dataloader, epoch, optimizer, scheduler, args):
    print("Start validating epoch {} ......".format(epoch))

    global use_gpu, best_top1_acc

    top1_accs = []
    backbone.eval()
    cls_head.eval()
    with torch.no_grad():
        preds = []
        gt_labels = []
        validate_start_time = time.time()
        if action_number_choosing:
            for _, (data, targets) in enumerate(val_dataloader, 0):
                pred, gt_label = helper.backbone_forward_test(backbone, cls_head, data, targets, top1_accs)
                preds.extend(pred)
                gt_labels.extend(gt_label)
        else:
            for _, data in enumerate(val_dataloader, 0):
                pred, gt_label = helper.backbone_forward_test_single(backbone, cls_head, data, top1_accs)
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
            print('-----New best cls found!-----')
            helper.save_cls_outputs(preds, gt_labels, args)
            helper.save_cls_checkpoint(backbone, cls_head, optimizer, scheduler, epoch, best_top1_acc, 'best', args)
            
            
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
    backbone, cls_head = builder.model_builder(args)
    if ckpt_path:
        builder.load_cls_model(backbone, cls_head, ckpt_path)
    else:
        builder.load_cls_model(backbone, cls_head, args.ckpts)
    if use_gpu:
        backbone.cuda()
        cls_head.cuda()
        torch.backends.cudnn.benchmark = True
    # DP
    backbone = nn.DataParallel(backbone)
    cls_head = nn.DataParallel(cls_head)
    test(backbone, cls_head, test_dataloader, args)

def test(backbone, cls_head,  test_dataloader, args):
    global use_gpu
    top1_accs = []
    backbone.eval()
    cls_head.eval()
    preds = []
    gt_labels = []
    with torch.no_grad():
        test_start_time = time.time()
        if action_number_choosing:
            for _, (data, target) in enumerate(test_dataloader, 0):
                pred, gt_label = helper.backbone_forward_test(backbone, cls_head, data, target, top1_accs)
                preds.extend(pred)
                gt_labels.extend(gt_label)
        else:
            for _, data in enumerate(test_dataloader, 0):
                pred, gt_label = helper.backbone_forward_test_single(backbone, cls_head, data, top1_accs)
                preds.extend(pred)
                gt_labels.extend(gt_label)
                
    # evaluation results
    test_end_time = time.time()
    top1_acc = sum(top1_accs) / len(test_dataloader)
    test_time = test_end_time - test_start_time

    helper.save_cls_outputs(preds, gt_labels, args)
    print('save outputs success!')
    print('[Time: %s][TEST] test time: %dmin%ds, top1_acc: %.4f' % 
          (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), test_time//60, test_time%60, top1_acc))

