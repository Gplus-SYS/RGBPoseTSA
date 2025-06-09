import numpy as np
import random
import os
import torch
import torch.nn as nn
from datetime import datetime
from tools import builder
from tools.ps import helper_ps as helper
from utils import misc
from torch.cuda.amp import GradScaler
import time

def train_net(args):
    global action_number_choosing, use_gpu
    action_number_choosing = args.action_number_choosing
    use_gpu = torch.cuda.is_available()
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
    # loss
    bce = nn.BCEWithLogitsLoss().cuda()
    # build model
    backbone, ps_net = builder.model_builder(args)
    # build opti
    optimizer, scheduler = builder.optimizer_scheduler_builder([backbone, ps_net],args)
    if use_gpu:
        backbone.cuda()
        ps_net.cuda()


    # Hyperparameter
    start_epoch = 0
    global epoch_best_tas, pred_tious_best_5, pred_tious_best_75
    epoch_best_tas = 0
    pred_tious_best_5 = 0
    pred_tious_best_75 = 0



    # resume ckptsQ
    if args.resume:
        start_epoch, pred_tious_best_5, pred_tious_best_75 = builder.resume_ps_train(backbone, ps_net, optimizer, scheduler, args)
        print('resume ckpts @ %d epoch(pred_tious_best_5 = %.4f , pred_tious_best_75 = %.4f)'
              % (start_epoch - 1, pred_tious_best_5, pred_tious_best_75))
        
    # DP
    backbone = nn.DataParallel(backbone)
    ps_net = nn.DataParallel(ps_net)
    scaler = GradScaler()



    for epoch in range(start_epoch, args.total_epochs):
        epoch_start_time = time.time()
        pred_tious_5 = []
        pred_tious_75 = []

        backbone.train()
        ps_net.train()

        if args.fix_bn:
            backbone.apply(misc.fix_bn)
        
        # training epoch
        if action_number_choosing:
            for idx, (data, target) in enumerate(train_dataloader):
                helper.backbone_forward_train(backbone, ps_net, bce, data, target, optimizer, epoch, idx+1,
                                              len(train_dataloader), scaler, pred_tious_5, pred_tious_75, args)
        else:
            for idx, data in enumerate(train_dataloader):
                helper.backbone_forward_train_single(backbone, ps_net, bce, data, optimizer, epoch, idx+1,
                                              len(train_dataloader), scaler, pred_tious_5, pred_tious_75, args)
        # evaluation results
        pred_tious_mean_5 = sum(pred_tious_5) / len(train_dataset)
        pred_tious_mean_75 = sum(pred_tious_75) / len(train_dataset)
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        print('[Time: %s][Training] EPOCH: %d, epoch time: %dmin%ds, tIoU_5: %.4f, tIoU_75: %.4f'
              % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, epoch_time//60, epoch_time%60, 
                 pred_tious_mean_5, pred_tious_mean_75))
        

        validate_net(backbone, ps_net, val_dataloader, epoch, optimizer, scheduler, args)
        helper.save_checkpoint(backbone, ps_net, optimizer, scheduler, epoch, pred_tious_mean_5, pred_tious_mean_75, 'last', args)
        
        print('[VAL] EPOCH_BEST_TAS: %d, best tIoU_5: %.6f, best tIoU_75: %.6f' % (epoch_best_tas,
                                                                        pred_tious_best_5, pred_tious_best_75))
        
        # scheduler lr
        scheduler_cfg = args.lr_config
        scheduler_enable = scheduler_cfg.get('enable')
        if scheduler_enable:
            scheduler.step()

    print('[Time: %s][Testing last]' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    test_net(args, os.path.join(args.experiment_path, 'last.pth'))
    print('[Time: %s][Testing best]' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    test_net(args, os.path.join(args.experiment_path, 'best.pth'))

def validate_net(backbone, ps_net, val_dataloader, epoch, optimizer, scheduler, args):
    print("Start validating epoch {} ......".format(epoch))

    global use_gpu
    global epoch_best_tas, pred_tious_best_5, pred_tious_best_75

    transits_preds = []
    transits_labels = []
    pred_tious_test_5 = []
    pred_tious_test_75 = []

    backbone.eval()
    ps_net.eval()


    with torch.no_grad():
        validate_start_time = time.time()
        if action_number_choosing:
            for _, (data, targets) in enumerate(val_dataloader, 0):
                transits_pred, transits_label = helper.backbone_forward_test(backbone, ps_net, data, targets,
                                                 pred_tious_test_5, pred_tious_test_75, args)
                transits_preds.extend(transits_pred)
                transits_labels.extend(transits_label)
        else:
            for _, data in enumerate(val_dataloader, 0):
                transits_pred, transits_label = helper.backbone_forward_test_single(backbone, ps_net, data,
                                                 pred_tious_test_5, pred_tious_test_75, args)
                transits_preds.extend(transits_pred)
                transits_labels.extend(transits_label)
        # evaluation results
        validate_end_time = time.time()
        pred_tious_test_mean_5 = sum(pred_tious_test_5) / (len(val_dataloader))
        pred_tious_test_mean_75 = sum(pred_tious_test_75) / (len(val_dataloader))
        validate_time = validate_end_time - validate_start_time

        if pred_tious_test_mean_5 > pred_tious_best_5:
            pred_tious_best_5 = pred_tious_test_mean_5
        if pred_tious_test_mean_75 > pred_tious_best_75:
            pred_tious_best_75 = pred_tious_test_mean_75
            epoch_best_tas = epoch
            helper.save_outputs(transits_preds, transits_labels, args)
            helper.save_checkpoint(backbone, ps_net, optimizer, scheduler, epoch,
                                   pred_tious_best_5, pred_tious_best_75, 'best', args)

        print('[Time: %s][Validating] EPOCH: %d, validate time: %dmin%ds, tIoU_5: %.6f, tIoU_75: %.6f' % 
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, validate_time//60, validate_time%60, pred_tious_test_mean_5, pred_tious_test_mean_75))
            
def test_net(args, ckpt_path = None):
    print('Tester start ... ')
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
    backbone, ps_net = builder.model_builder(args)
    if ckpt_path:
        builder.load_ps_model(backbone, ps_net, ckpt_path)
    else:
        builder.load_ps_model(backbone, ps_net, args.ckpts)
    if use_gpu:
        backbone.cuda()
        ps_net.cuda()
        torch.backends.cudnn.benchmark = True
    # DP
    backbone = nn.DataParallel(backbone)
    ps_net = nn.DataParallel(ps_net)
    test(backbone, ps_net, test_dataloader, args)

def test(backbone, ps_net, test_dataloader, args):
    global use_gpu

    transits_preds = []
    transits_labels = []
    pred_tious_test_5 = []
    pred_tious_test_75 = []

    backbone.eval()
    ps_net.eval()

    with torch.no_grad():
        test_start_time = time.time()
        if action_number_choosing:
            for _, (data, target) in enumerate(test_dataloader, 0):
                transits_pred, transits_label = helper.backbone_forward_test(backbone, ps_net, data, target,
                                                 pred_tious_test_5, pred_tious_test_75, args)
                transits_preds.extend(transits_pred)
                transits_labels.extend(transits_label)
        else:
            for _, data in enumerate(test_dataloader, 0):
                transits_pred, transits_label = helper.backbone_forward_test_single(backbone, ps_net, data,
                                                 pred_tious_test_5, pred_tious_test_75, args)
                transits_preds.extend(transits_pred)
                transits_labels.extend(transits_label)

    # evaluation results
    test_end_time = time.time()
    pred_tious_test_mean_5 = sum(pred_tious_test_5) / (len(test_dataloader))
    pred_tious_test_mean_75 = sum(pred_tious_test_75) / (len(test_dataloader))

    test_time = test_end_time - test_start_time
    
    helper.save_outputs(transits_preds, transits_labels, args)
    print('save outputs success!')
    print('[Time: %s][TEST] test time: %dmin%ds' % 
          (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), test_time//60, test_time%60))
    print('[Time: %s][TEST] tIoU_5: %.6f, tIoU_75: %.6f' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pred_tious_test_mean_5, pred_tious_test_mean_75))