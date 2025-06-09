import numpy as np
import random
import os
import torch
import torch.nn as nn
from datetime import datetime
from tools import builder
from tools.aqa import helper_aqa_MM_late_fusion as helper
from utils import misc
from scipy import stats
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
    mse = nn.MSELoss().cuda()
    bce = nn.BCEWithLogitsLoss().cuda()
    # build model
    rgb_backbone, pose_backbone, ps_net, decoder, regressor = builder.model_builder(args)
    # build opti
    optimizer, scheduler = builder.optimizer_scheduler_builder([rgb_backbone, pose_backbone, ps_net, decoder, regressor],args)
    if use_gpu:
        rgb_backbone.cuda()
        pose_backbone.cuda()
        ps_net.cuda()
        decoder.cuda()
        regressor.cuda()

    # Hyperparameter
    start_epoch = 0
    global epoch_best_tas, pred_tious_best_5, pred_tious_best_75, epoch_best_aqa, rho_best, L2_min, RL2_min
    epoch_best_tas = 0
    pred_tious_best_5 = 0
    pred_tious_best_75 = 0
    epoch_best_aqa = 0
    rho_best = 0
    L2_min = 1000
    RL2_min = 1000


    # resume ckptsQ
    if args.resume:
        start_epoch, epoch_best_aqa, rho_best, L2_min, RL2_min = builder.MM_resume_train(rgb_backbone, pose_backbone, ps_net, decoder, 
                                                                                      regressor, optimizer, scheduler, args)
        print('resume ckpts @ %d epoch(rho = %.4f, L2 = %.4f , RL2 = %.4f)'
              % (start_epoch - 1, rho_best, L2_min, RL2_min))
        
    # DP
    rgb_backbone = nn.DataParallel(rgb_backbone)
    pose_backbone = nn.DataParallel(pose_backbone)
    ps_net = nn.DataParallel(ps_net)
    decoder = nn.DataParallel(decoder)
    regressor = nn.DataParallel(regressor)
    scaler = GradScaler()



    for epoch in range(start_epoch, args.total_epochs):
        epoch_start_time = time.time()
        pred_tious_5 = []
        pred_tious_75 = []
        true_scores = []
        pred_scores = []

    
        rgb_backbone.train()
        pose_backbone.train()
        ps_net.train()
        decoder.train()
        regressor.train()
        if args.fix_bn:
            rgb_backbone.apply(misc.fix_bn)
            pose_backbone.apply(misc.fix_bn)
        
        # training epoch
        if action_number_choosing:
            for idx, (data, target) in enumerate(train_dataloader):
                helper.backbone_forward_train(rgb_backbone, pose_backbone, ps_net, decoder, regressor, bce, mse, 
                                                    data, target, optimizer, epoch, idx+1, len(train_dataloader), 
                                                    scaler, pred_scores, pred_tious_5, pred_tious_75, args)
                true_scores.extend(data['dive_score'].reshape(-1).numpy())

        # evaluation results
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]
        pred_tious_mean_5 = sum(pred_tious_5) / len(train_dataset)
        pred_tious_mean_75 = sum(pred_tious_75) / len(train_dataset)
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        print('[Time: %s][Training] EPOCH: %d, epoch time: %dmin%ds, tIoU_5: %.4f, tIoU_75: %.4f, correlation: %.4f, L2: %.4f, RL2: %.4f'
              % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, epoch_time//60, epoch_time%60, 
                 pred_tious_mean_5, pred_tious_mean_75, rho, L2, RL2))
        

        validate_net(rgb_backbone, pose_backbone, ps_net, decoder, regressor, val_dataloader, epoch, optimizer, scheduler, args)
        helper.save_checkpoint(rgb_backbone, pose_backbone, ps_net, decoder, regressor, optimizer, scheduler, epoch, 
                                      epoch_best_aqa, rho_best, L2_min, RL2_min, 'last', args)
        
        print('[VAL] EPOCH_BEST_TAS: %d, best tIoU_5: %.6f, best tIoU_75: %.6f' % (epoch_best_tas,
                                                                        pred_tious_best_5, pred_tious_best_75))
        print('[VAL] EPOCH_BEST_AQA: %d, best correlation: %.6f, best L2: %.6f, best RL2: %.6f' % (epoch_best_aqa,
                                                                                        rho_best, L2_min, RL2_min))
        
        # scheduler lr
        scheduler_cfg = args.lr_config
        scheduler_enable = scheduler_cfg.get('enable')
        if scheduler_enable:
            scheduler.step()

    print('[Time: %s][Testing last]' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    test_net(args, os.path.join(args.experiment_path, 'last.pth'))
    print('[Time: %s][Testing best]' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    test_net(args, os.path.join(args.experiment_path, 'best.pth'))

def validate_net(rgb_backbone, pose_backbone, ps_net, decoder, regressor, val_dataloader, epoch, optimizer, scheduler, args):
    print("Start validating epoch {} ......".format(epoch))

    global use_gpu
    global epoch_best_tas, rho_best, L2_min, RL2_min, epoch_best_aqa, pred_tious_best_5, pred_tious_best_75

    true_scores = []
    pred_scores = []
    transits_preds = []
    transits_labels = []
    pred_tious_test_5 = []
    pred_tious_test_75 = []

    rgb_backbone.eval()
    pose_backbone.eval()
    ps_net.eval()
    decoder.eval()
    regressor.eval()

    with torch.no_grad():
        validate_start_time = time.time()
        if action_number_choosing:
            for _, (data, targets) in enumerate(val_dataloader, 0):
                transits_pred, transits_label = helper.backbone_forward_test(rgb_backbone, pose_backbone, ps_net, decoder, regressor, data, targets, 
                                                 pred_scores, pred_tious_test_5, pred_tious_test_75, args)
                transits_preds.extend(transits_pred)
                transits_labels.extend(transits_label)
                true_scores.extend(data['dive_score'].reshape(-1).numpy())

        # evaluation results
        validate_end_time = time.time()
        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, p = stats.spearmanr(pred_scores, true_scores)
        L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
        RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]
        pred_tious_test_mean_5 = sum(pred_tious_test_5) / (len(val_dataloader))
        pred_tious_test_mean_75 = sum(pred_tious_test_75) / (len(val_dataloader))
        validate_time = validate_end_time - validate_start_time

        if pred_tious_test_mean_5 > pred_tious_best_5:
            pred_tious_best_5 = pred_tious_test_mean_5
        if pred_tious_test_mean_75 > pred_tious_best_75:
            pred_tious_best_75 = pred_tious_test_mean_75
            epoch_best_tas = epoch

        print('[Time: %s][Validating] EPOCH: %d, validate time: %dmin%ds, tIoU_5: %.6f, tIoU_75: %.6f' % 
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, validate_time//60, validate_time%60, pred_tious_test_mean_5, pred_tious_test_mean_75))
        
        if L2_min > L2:
            L2_min = L2
        if RL2_min > RL2:
            RL2_min = RL2
        if rho > rho_best:
            rho_best = rho
            epoch_best_aqa = epoch
            print('-----New best found!-----')
            helper.save_outputs(pred_scores, true_scores, transits_preds, transits_labels, args)

            helper.save_checkpoint(rgb_backbone, pose_backbone, ps_net, decoder, regressor,
                                        optimizer, scheduler, epoch, epoch_best_aqa,
                                        rho_best, L2_min, RL2_min, 'best', args)
            
        print('[Time: %s][Validating] EPOCH: %d, correlation: %.6f, L2: %.6f, RL2: %.6f' % 
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, rho, L2, RL2))
            
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
    rgb_backbone, pose_backbone, ps_net, decoder, regressor = builder.model_builder(args)
    if ckpt_path:
        builder.MM_load_model(rgb_backbone, pose_backbone, ps_net, decoder, regressor, ckpt_path)
    else:
        builder.MM_load_model(rgb_backbone, pose_backbone, ps_net, decoder, regressor, args.ckpts)
    if use_gpu:
        rgb_backbone.cuda()
        pose_backbone.cuda()
        ps_net.cuda()
        decoder.cuda()
        regressor.cuda()
        torch.backends.cudnn.benchmark = True
    # DP
    rgb_backbone = nn.DataParallel(rgb_backbone)
    pose_backbone = nn.DataParallel(pose_backbone)
    ps_net = nn.DataParallel(ps_net)
    decoder = nn.DataParallel(decoder)
    regressor = nn.DataParallel(regressor)
    test(rgb_backbone, pose_backbone, ps_net, decoder, regressor, test_dataloader, args)

def test(rgb_backbone, pose_backbone, ps_net, decoder, regressor, test_dataloader, args):
    global use_gpu
    true_scores = []
    pred_scores = []
    transits_preds = []
    transits_labels = []
    pred_tious_test_5 = []
    pred_tious_test_75 = []

    rgb_backbone.eval()
    pose_backbone.eval()
    ps_net.eval()
    decoder.eval()
    regressor.eval()
    with torch.no_grad():
        test_start_time = time.time()
        if action_number_choosing:
            for _, (data, target) in enumerate(test_dataloader, 0):
                transits_pred, transits_label = helper.backbone_forward_test(rgb_backbone, pose_backbone, ps_net, decoder, regressor, data, target, pred_scores, 
                                                 pred_tious_test_5, pred_tious_test_75, args)
                transits_preds.extend(transits_pred)
                transits_labels.extend(transits_label)
                true_scores.extend(data['dive_score'].reshape(-1).numpy())

    # evaluation results
    test_end_time = time.time()
    pred_scores = np.array(pred_scores)
    true_scores = np.array(true_scores)
    rho, p = stats.spearmanr(pred_scores, true_scores)
    L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
    RL2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]
    pred_tious_test_mean_5 = sum(pred_tious_test_5) / (len(test_dataloader))
    pred_tious_test_mean_75 = sum(pred_tious_test_75) / (len(test_dataloader))

    test_time = test_end_time - test_start_time

    helper.save_outputs(pred_scores, true_scores, transits_preds, transits_labels, args)
    print('save outputs success!')
    print('[Time: %s][TEST] test time: %dmin%ds' % 
          (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), test_time//60, test_time%60))
    print('[Time: %s][TEST] tIoU_5: %.6f, tIoU_75: %.6f' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pred_tious_test_mean_5, pred_tious_test_mean_75))
    print('[Time: %s][TEST] correlation: %.6f, L2: %.6f, RL2: %.6f' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  rho, L2, RL2))
   