import numpy as np
import random
import os
import torch
import torch.nn as nn
from datetime import datetime
from tools import builder
from tools.aqa import helper_aqa_MM_early_fusion_double as helper
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
    backbone, ps_net_rgb, ps_net_pose, decoder_rgb, decoder_pose, regressor_rgb, regressor_pose = builder.model_builder(args)
    # build opti
    optimizer, scheduler = builder.optimizer_scheduler_builder([backbone, ps_net_rgb, ps_net_pose, decoder_rgb, decoder_pose, regressor_rgb, regressor_pose],args)
    if use_gpu:
        backbone.cuda()
        ps_net_rgb.cuda()
        ps_net_pose.cuda()
        decoder_rgb.cuda()
        decoder_pose.cuda()
        regressor_rgb.cuda()
        regressor_pose.cuda()

    # Hyperparameter
    start_epoch = 0
    global epoch_best_tas_rgb, pred_tious_best_5_rgb, pred_tious_best_75_rgb, epoch_best_aqa_rgb, rho_best_rgb, L2_min_rgb, RL2_min_rgb
    epoch_best_tas_rgb = 0
    pred_tious_best_5_rgb = 0
    pred_tious_best_75_rgb = 0
    epoch_best_aqa_rgb = 0
    rho_best_rgb = 0
    L2_min_rgb = 1000
    RL2_min_rgb = 1000
    global epoch_best_tas_pose, pred_tious_best_5_pose, pred_tious_best_75_pose, epoch_best_aqa_pose, rho_best_pose, L2_min_pose, RL2_min_pose
    epoch_best_tas_pose = 0
    pred_tious_best_5_pose = 0
    pred_tious_best_75_pose = 0
    epoch_best_aqa_pose = 0
    rho_best_pose = 0
    L2_min_pose = 1000
    RL2_min_pose = 1000


    # resume ckptsQ
    if args.resume:
        start_epoch, epoch_best_aqa_rgb, epoch_best_aqa_pose, rho_best_rgb, 
        rho_best_pose, L2_min_rgb, L2_min_pose, RL2_min_rgb, RL2_min_pose = builder.resume_train_double_path(backbone, ps_net_rgb, ps_net_pose, 
                                                                                                 decoder_rgb, decoder_pose, regressor_rgb, 
                                                                                                 regressor_pose, optimizer, scheduler, args)
        print('resume ckpts @ %d epoch(rho_rgb = %.4f, L2_rgb = %.4f , RL2_rgb = %.4f)(rho_pose = %.4f, L2_pose = %.4f , RL2_pose = %.4f)'
              % (start_epoch - 1, rho_best_rgb, L2_min_rgb, RL2_min_rgb, rho_best_pose, L2_min_pose, RL2_min_pose))
        
    # DP
    backbone = nn.DataParallel(backbone)
    ps_net_rgb = nn.DataParallel(ps_net_rgb)
    ps_net_pose = nn.DataParallel(ps_net_pose)
    decoder_rgb = nn.DataParallel(decoder_rgb)
    decoder_pose = nn.DataParallel(decoder_pose)
    regressor_rgb = nn.DataParallel(regressor_rgb)
    regressor_pose = nn.DataParallel(regressor_pose)
    scaler = GradScaler()



    for epoch in range(start_epoch, args.total_epochs):
        epoch_start_time = time.time()
        pred_tious_5_rgb = []
        pred_tious_5_pose = []
        pred_tious_75_rgb = []
        pred_tious_75_pose = []
        true_scores = []
        pred_scores_rgb = []
        pred_scores_pose = []


    
        backbone.train()
        ps_net_rgb.train()
        ps_net_pose.train()
        decoder_rgb.train()
        decoder_pose.train()
        regressor_rgb.train()
        regressor_pose.train()
        if args.fix_bn:
            backbone.module.rgb_path.apply(misc.fix_bn)
            backbone.module.pose_path.apply(misc.fix_bn)
            print('The batchnorm layer of rgb path and pose path has been frozen!')
        
        # training epoch
        if action_number_choosing:
            for idx, (data, target) in enumerate(train_dataloader):
                helper.backbone_forward_train(backbone, ps_net_rgb, ps_net_pose, decoder_rgb, decoder_pose, 
                                                                  regressor_rgb, regressor_pose, bce, mse, data, target, optimizer, 
                                                                  epoch, idx+1, len(train_dataloader), scaler, pred_scores_rgb, pred_scores_pose,
                                                                  pred_tious_5_rgb, pred_tious_5_pose, pred_tious_75_rgb, pred_tious_75_pose, args)
                true_scores.extend(data['dive_score'].reshape(-1).numpy())

        # evaluation results
        pred_scores_rgb = np.array(pred_scores_rgb)
        pred_scores_pose = np.array(pred_scores_pose)
        true_scores = np.array(true_scores)
        rho_rgb, p_rgb = stats.spearmanr(pred_scores_rgb, true_scores)
        rho_pose, p_pose = stats.spearmanr(pred_scores_pose, true_scores)
        L2_rgb = np.power(pred_scores_rgb - true_scores, 2).sum() / true_scores.shape[0]
        L2_pose = np.power(pred_scores_pose - true_scores, 2).sum() / true_scores.shape[0]
        RL2_rgb = np.power((pred_scores_rgb - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]
        RL2_pose = np.power((pred_scores_pose - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]
        pred_tious_mean_5_rgb = sum(pred_scores_rgb) / len(train_dataset)
        pred_tious_mean_5_pose = sum(pred_scores_pose) / len(train_dataset)
        pred_tious_mean_75_rgb = sum(pred_tious_75_rgb) / len(train_dataset)
        pred_tious_mean_75_pose = sum(pred_tious_75_pose) / len(train_dataset)
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        print('[Time: %s][Training][RGB Path] EPOCH: %d, epoch time: %dmin%ds, tIoU_5: %.4f, tIoU_75: %.4f, correlation: %.4f, L2: %.4f, RL2: %.4f'
              % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, epoch_time//60, epoch_time%60, 
                 pred_tious_mean_5_rgb, pred_tious_mean_75_rgb, rho_rgb, L2_rgb, RL2_rgb))
        print('[Time: %s][Training][Pose Path] EPOCH: %d, epoch time: %dmin%ds, tIoU_5: %.4f, tIoU_75: %.4f, correlation: %.4f, L2: %.4f, RL2: %.4f'
              % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, epoch_time//60, epoch_time%60, 
                 pred_tious_mean_5_pose, pred_tious_mean_75_pose, rho_pose, L2_pose, RL2_pose))

        validate_net(backbone, ps_net_rgb, ps_net_pose, decoder_rgb, decoder_pose, regressor_rgb, regressor_pose, \
                     val_dataloader, epoch, optimizer, scheduler, args)
        helper.save_checkpoint(backbone, ps_net_rgb, ps_net_pose, decoder_rgb, decoder_pose, 
                                                   regressor_rgb, regressor_pose, optimizer, scheduler, epoch, 
                                                   epoch_best_aqa_rgb, epoch_best_aqa_pose, rho_best_rgb, rho_best_pose, 
                                                   L2_min_rgb, L2_min_pose, RL2_min_rgb, RL2_min_pose, 'last', args)
        
        print('[VAL][RGB Path] EPOCH_BEST_TAS: %d, best tIoU_5: %.6f, best tIoU_75: %.6f' % (epoch_best_tas_rgb,
                                                                        pred_tious_best_5_rgb, pred_tious_best_75_rgb))
        print('[VAL][RGB Path] EPOCH_BEST_AQA: %d, best correlation: %.6f, best L2: %.6f, best RL2: %.6f' % (epoch_best_aqa_rgb,
                                                                                        rho_best_rgb, L2_min_rgb, RL2_min_rgb))
        print('[VAL][Pose Path] EPOCH_BEST_TAS: %d, best tIoU_5: %.6f, best tIoU_75: %.6f' % (epoch_best_tas_pose,
                                                                        pred_tious_best_5_pose, pred_tious_best_75_pose))
        print('[VAL][Pose Path] EPOCH_BEST_AQA: %d, best correlation: %.6f, best L2: %.6f, best RL2: %.6f' % (epoch_best_aqa_pose,
                                                                                        rho_best_pose, L2_min_pose, RL2_min_pose))
        
        # scheduler lr
        scheduler_cfg = args.lr_config
        scheduler_enable = scheduler_cfg.get('enable')
        if scheduler_enable:
            scheduler.step()

    print('[Time: %s][Testing last]' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    test_net(args, os.path.join(args.experiment_path, 'last.pth'))
    print('[Time: %s][Testing best RGB Path]' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    test_net(args, os.path.join(args.experiment_path, 'best_rgb.pth'))
    print('[Time: %s][Testing best Pose Path]' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    test_net(args, os.path.join(args.experiment_path, 'best_pose.pth'))

def validate_net(backbone, ps_net_rgb, ps_net_pose, decoder_rgb, decoder_pose, regressor_rgb, regressor_pose,
                  val_dataloader, epoch, optimizer, scheduler, args):
    print("Start validating epoch {} ......".format(epoch))

    global use_gpu
    global epoch_best_tas_rgb, pred_tious_best_5_rgb, pred_tious_best_75_rgb, epoch_best_aqa_rgb, rho_best_rgb, L2_min_rgb, RL2_min_rgb
    global epoch_best_tas_pose, pred_tious_best_5_pose, pred_tious_best_75_pose, epoch_best_aqa_pose, rho_best_pose, L2_min_pose, RL2_min_pose
    
    pred_tious_test_5_rgb = []
    pred_tious_test_5_pose = []
    pred_tious_test_75_rgb = []
    pred_tious_test_75_pose = []
    true_scores = []
    pred_scores_rgb = []
    pred_scores_pose = []
    transits_labels = []
    transits_preds_rgb = []
    transits_preds_pose = []

    backbone.eval()
    ps_net_rgb.eval()
    ps_net_pose.eval()
    decoder_rgb.eval()
    decoder_pose.eval()
    regressor_rgb.eval()
    regressor_pose.eval()

    with torch.no_grad():
        validate_start_time = time.time()
        if action_number_choosing:
            for _, (data, targets) in enumerate(val_dataloader, 0):
                t_p_rgb, t_p_pose, transits_label = helper.backbone_forward_test(backbone, ps_net_rgb, ps_net_pose, decoder_rgb, 
                                                                                                       decoder_pose, regressor_rgb, regressor_pose, data, 
                                                                                                       targets, pred_scores_rgb, pred_scores_pose, 
                                                                                                       pred_tious_test_5_rgb, pred_tious_test_5_pose, 
                                                                                                       pred_tious_test_75_rgb, pred_tious_test_75_pose, args)
                transits_preds_rgb.extend(t_p_rgb)
                transits_preds_pose.extend(t_p_pose)
                transits_labels.extend(transits_label)
                true_scores.extend(data['dive_score'].reshape(-1).numpy())

        # evaluation results
        validate_end_time = time.time()
        pred_scores_rgb = np.array(pred_scores_rgb)
        pred_scores_pose = np.array(pred_scores_pose)
        true_scores = np.array(true_scores)
        rho_rgb, p_rgb = stats.spearmanr(pred_scores_rgb, true_scores)
        rho_pose, p_pose = stats.spearmanr(pred_scores_pose, true_scores)
        L2_rgb = np.power(pred_scores_rgb - true_scores, 2).sum() / true_scores.shape[0]
        L2_pose = np.power(pred_scores_pose - true_scores, 2).sum() / true_scores.shape[0]
        RL2_rgb = np.power((pred_scores_rgb - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]
        RL2_pose = np.power((pred_scores_pose - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]
        pred_tious_test_mean_5_rgb = sum(pred_tious_test_5_rgb) / (len(val_dataloader))
        pred_tious_test_mean_5_pose = sum(pred_tious_test_5_pose) / (len(val_dataloader))
        pred_tious_test_mean_75_rgb = sum(pred_tious_test_75_rgb) / (len(val_dataloader))
        pred_tious_test_mean_75_pose = sum(pred_tious_test_75_pose) / (len(val_dataloader))
        validate_time = validate_end_time - validate_start_time

        if pred_tious_test_mean_5_rgb > pred_tious_best_5_rgb:
            pred_tious_best_5_rgb = pred_tious_test_mean_5_rgb
        if pred_tious_test_mean_5_pose > pred_tious_best_5_pose:
            pred_tious_best_5_pose = pred_tious_test_mean_5_pose   
        if pred_tious_test_mean_75_rgb > pred_tious_best_75_rgb:
            pred_tious_best_75_rgb = pred_tious_test_mean_75_rgb
            epoch_best_tas_rgb = epoch
        if pred_tious_test_mean_75_pose > pred_tious_best_75_pose:
            pred_tious_best_75_pose = pred_tious_test_mean_75_pose
            epoch_best_tas_pose = epoch
        print('[Time: %s][Validating][RGB Path] EPOCH: %d, validate time: %dmin%ds, tIoU_5: %.6f, tIoU_75: %.6f' % 
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, validate_time//60, validate_time%60, pred_tious_test_mean_5_rgb, pred_tious_test_mean_75_rgb))
        print('[Time: %s][Validating][Pose Path] EPOCH: %d, validate time: %dmin%ds, tIoU_5: %.6f, tIoU_75: %.6f' % 
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, validate_time//60, validate_time%60, pred_tious_test_mean_5_pose, pred_tious_test_mean_75_pose))
        if L2_min_rgb > L2_rgb:
            L2_min_rgb = L2_rgb
        if L2_min_pose > L2_pose:
            L2_min_pose = L2_pose
        if RL2_min_rgb > RL2_rgb:
            RL2_min_rgb = RL2_rgb
        if RL2_min_pose > RL2_pose:
            RL2_min_pose = RL2_pose
        if rho_rgb > rho_best_rgb:
            rho_best_rgb = rho_rgb
            epoch_best_aqa_rgb = epoch
            print('-----RGB Path New best found!-----')
            helper.save_outputs(pred_scores_rgb, true_scores, transits_preds_rgb, transits_labels,'RGB_train', args)

            helper.save_checkpoint(backbone, ps_net_rgb, ps_net_pose, decoder_rgb, decoder_pose, regressor_rgb, regressor_pose,
                                   optimizer, scheduler, epoch, epoch_best_aqa_rgb, epoch_best_aqa_pose, rho_best_rgb, rho_best_pose, 
                                   L2_min_rgb, L2_min_pose, RL2_min_rgb, RL2_min_pose, 'best_rgb', args)
        
        if rho_pose > rho_best_pose:
            rho_best_pose = rho_pose
            epoch_best_aqa_pose = epoch
            print('-----Pose Path New best found!-----')
            helper.save_outputs(pred_scores_pose, true_scores, transits_preds_pose, transits_labels, 'Pose_train', args)
            helper.save_checkpoint(backbone, ps_net_rgb, ps_net_pose, decoder_rgb, decoder_pose, regressor_rgb, regressor_pose,
                                   optimizer, scheduler, epoch, epoch_best_aqa_rgb, epoch_best_aqa_pose, rho_best_rgb, rho_best_pose, 
                                   L2_min_rgb, L2_min_pose, RL2_min_rgb, RL2_min_pose, 'best_pose', args)
            
        print('[Time: %s][Validating][RGB Path] EPOCH: %d, correlation: %.6f, L2: %.6f, RL2: %.6f' % 
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, rho_rgb, L2_rgb, RL2_rgb))
        print('[Time: %s][Validating][Pose Path] EPOCH: %d, correlation: %.6f, L2: %.6f, RL2: %.6f' % 
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, rho_pose, L2_pose, RL2_pose))
            
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
    backbone, ps_net_rgb, ps_net_pose, decoder_rgb, decoder_pose, regressor_rgb, regressor_pose = builder.model_builder(args)
    if ckpt_path:
        builder.load_model_double_path(backbone, ps_net_rgb, ps_net_pose, decoder_rgb, decoder_pose, regressor_rgb, regressor_pose, ckpt_path)
    else:
        builder.load_model_double_path(backbone, ps_net_rgb, ps_net_pose, decoder_rgb, decoder_pose, regressor_rgb, regressor_pose, args.ckpts)
    if use_gpu:
        backbone.cuda()
        ps_net_rgb.cuda()
        ps_net_pose.cuda()
        decoder_rgb.cuda()
        decoder_pose.cuda()
        regressor_rgb.cuda()
        regressor_pose.cuda()
        torch.backends.cudnn.benchmark = True
    # DP
    backbone = nn.DataParallel(backbone)
    ps_net_rgb = nn.DataParallel(ps_net_rgb)
    ps_net_pose = nn.DataParallel(ps_net_pose)
    decoder_rgb = nn.DataParallel(decoder_rgb)
    decoder_pose = nn.DataParallel(decoder_pose)
    regressor_rgb = nn.DataParallel(regressor_rgb)
    regressor_pose = nn.DataParallel(regressor_pose)

    test(backbone, ps_net_rgb, ps_net_pose, decoder_rgb, decoder_pose, regressor_rgb, regressor_pose, test_dataloader, args)

def test(backbone, ps_net_rgb, ps_net_pose, decoder_rgb, decoder_pose, regressor_rgb, regressor_pose, test_dataloader, args):
    global use_gpu
    true_scores = []
    pred_scores_rgb = []
    pred_scores_pose = []
    transits_preds_rgb = []
    transits_preds_pose = []
    transits_labels = []
    pred_tious_test_5_rgb = []
    pred_tious_test_5_pose = []
    pred_tious_test_75_rgb = []
    pred_tious_test_75_pose = []

    backbone.eval()
    ps_net_rgb.eval()
    ps_net_pose.eval()
    decoder_rgb.eval()
    decoder_pose.eval()
    regressor_rgb.eval()
    regressor_pose.eval()
    with torch.no_grad():
        test_start_time = time.time()
        if action_number_choosing:
            for _, (data, target) in enumerate(test_dataloader, 0):
                t_p_rgb, t_p_pose, transits_label = helper.backbone_forward_test(backbone, ps_net_rgb, ps_net_pose, decoder_rgb, decoder_pose, 
                                                                             regressor_rgb, regressor_pose, data, target, pred_scores_rgb, 
                                                                             pred_scores_pose, pred_tious_test_5_rgb, pred_tious_test_5_pose, 
                                                                             pred_tious_test_75_rgb, pred_tious_test_75_pose, args)
                transits_preds_rgb.extend(t_p_rgb)
                transits_preds_pose.extend(t_p_pose)
                transits_labels.extend(transits_label)
                true_scores.extend(data['dive_score'].reshape(-1).numpy())

    # evaluation results
    test_end_time = time.time()
    pred_scores_rgb = np.array(pred_scores_rgb)
    pred_scores_pose = np.array(pred_scores_pose)
    true_scores = np.array(true_scores)
    rho_rgb, p_rgb = stats.spearmanr(pred_scores_rgb, true_scores)
    rho_pose, p_pose = stats.spearmanr(pred_scores_pose, true_scores)
    L2_rgb = np.power(pred_scores_rgb - true_scores, 2).sum() / true_scores.shape[0]
    L2_pose = np.power(pred_scores_pose - true_scores, 2).sum() / true_scores.shape[0]
    RL2_rgb = np.power((pred_scores_rgb - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]
    RL2_pose = np.power((pred_scores_pose - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]
    pred_tious_test_mean_5_rgb = sum(pred_tious_test_5_rgb) / (len(test_dataloader))
    pred_tious_test_mean_5_pose = sum(pred_tious_test_5_pose) / (len(test_dataloader))
    pred_tious_test_mean_75_rgb = sum(pred_tious_test_75_rgb) / (len(test_dataloader))
    pred_tious_test_mean_75_pose = sum(pred_tious_test_75_pose) / (len(test_dataloader))
    test_time = test_end_time - test_start_time

    helper.save_outputs(pred_scores_rgb, true_scores, transits_preds_rgb, transits_labels,'RGB_test', args)
    helper.save_outputs(pred_scores_pose, true_scores, transits_preds_pose, transits_labels, 'Pose_test', args)
    
    print('[Time: %s][TEST] test time: %dmin%ds' % 
          (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), test_time//60, test_time%60))
    print('[Time: %s][TEST][RGB Path] tIoU_5: %.6f, tIoU_75: %.6f' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pred_tious_test_mean_5_rgb, pred_tious_test_mean_75_rgb))
    print('[Time: %s][TEST][Pose Path] tIoU_5: %.6f, tIoU_75: %.6f' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pred_tious_test_mean_5_pose, pred_tious_test_mean_75_pose))
    print('[Time: %s][TEST][RGB Path] correlation: %.6f, L2: %.6f, RL2: %.6f' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  rho_rgb, L2_rgb, RL2_rgb))
    print('[Time: %s][TEST][Pose Path] correlation: %.6f, L2: %.6f, RL2: %.6f' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  rho_pose, L2_pose, RL2_pose))