import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../../"))

import torch
import torch.nn.functional as F
import numpy as np
from models.core import top_k_accuracy
from datetime import datetime
from torch.cuda.amp import autocast
from utils.misc import segment_iou, cal_tiou, seg_pool_1d, seg_pool_3d

def data_process_train(data, target):
    imgs_data = data['imgs'].cuda()
    label_data = data['label'].cuda()
    transits_data = data['transits'].cuda()
    dive_score_data = data['dive_score'].cuda()

    imgs_target = target['imgs'].cuda()
    label_target = target['label'].cuda()
    transits_target = target['transits'].cuda()
    dive_score_target = target['dive_score'].cuda()

    label = torch.cat((label_data, label_target), 0)
    # input shape is [batch, num_clips, frames, height, width]
    
    imgs_data = imgs_data.reshape((-1, ) + imgs_data.shape[2:])
    imgs_target = imgs_target.reshape((-1, ) + imgs_target.shape[2:])

    return label, imgs_data, imgs_target, transits_data, transits_target, dive_score_data, dive_score_target

def I3D_process(backbone, imgs_data, imgs_target):
    feature_1, feature_2, feamap_1, feamap_2, cls_feamap_12 = backbone([imgs_data, imgs_target])
    N, T, C, T_t, H_t, W_t = feamap_1.size()
    feamap_1 = feamap_1.mean(-3)
    feamap_2 = feamap_2.mean(-3)
    feamap_1_re = feamap_1.reshape(-1, T, C)
    feamap_2_re = feamap_2.reshape(-1, T, C)
    com_feature_12_u = torch.cat((feature_1, feature_2), 0)
    com_feamap_12_u = torch.cat((feamap_1_re, feamap_2_re), 0)
    return cls_feamap_12, com_feature_12_u, com_feamap_12_u, N, T, C, T_t, H_t, W_t


def ps_net_process_train(com_feature_12_u, com_feamap_12_u, ps_net, N, H, W, transits_data, transits_target, bce):
    u_fea_96, transits_pred = ps_net(com_feature_12_u)
    u_feamap_96, transits_pred_map = ps_net(com_feamap_12_u)
    u_feamap_96 = u_feamap_96.reshape(2*N, u_feamap_96.shape[1], u_feamap_96.shape[2], H, W)

    label_12_tas = torch.cat((transits_data, transits_target), 0)
    label_12_pad = torch.zeros(transits_pred.size())
    
    for bs in range(transits_pred.shape[0]):
        label_12_pad[bs, int(label_12_tas[bs, 0]), 0] = 1
        label_12_pad[bs, int(label_12_tas[bs, -1]), -1] = 1
    loss_tas = bce(transits_pred, label_12_pad.cuda())

    num = round(transits_pred.shape[1] / transits_pred.shape[-1])
    transits_st_ed = torch.zeros(label_12_tas.size())
    
    for bs in range(transits_pred.shape[0]):
        for i in range(transits_pred.shape[-1]):
            transits_st_ed[bs, i] = transits_pred[bs, i * num: (i + 1) * num, i].argmax(0).cpu().item() + i * num
        # first_transit_pred = transits_pred[bs, :-1, 0].argmax(0).cpu().item()
        # transits_st_ed[bs, 0] = first_transit_pred
        # second_transit_pred = transits_pred[bs, first_transit_pred + 1:, 1].argmax(0).cpu().item() + first_transit_pred + 1
        # transits_st_ed[bs, 1] = second_transit_pred

    label_1_tas_pred = transits_st_ed[:transits_st_ed.shape[0] // 2]
    label_2_tas_pred = transits_st_ed[transits_st_ed.shape[0] // 2:]

    return u_fea_96, transits_pred, u_feamap_96, label_12_tas, loss_tas, transits_st_ed, label_1_tas_pred, label_2_tas_pred

def decoder_process_train(u_fea_96, u_feamap_96, transits_data, transits_target, label_1_tas_pred, label_2_tas_pred, decoder, H, W, epoch, args):
    u_fea_96_1 = u_fea_96[:u_fea_96.shape[0] // 2].transpose(1, 2)
    u_fea_96_2 = u_fea_96[u_fea_96.shape[0] // 2:].transpose(1, 2)

    u_feamap_96_1 = u_feamap_96[:u_feamap_96.shape[0] // 2].transpose(1, 2)
    u_feamap_96_2 = u_feamap_96[u_feamap_96.shape[0] // 2:].transpose(1, 2)

    if epoch / args.total_epochs <= args.prob_tas_threshold:
        video_1_segs = []
        for bs_1 in range(u_fea_96_1.shape[0]):
            video_1_st = int(transits_data[bs_1][0].item())
            video_1_ed = int(transits_data[bs_1][1].item())
            video_1_segs.append(seg_pool_1d(u_fea_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
        video_1_segs = torch.cat(video_1_segs, 0).transpose(1, 2)

        video_2_segs = []
        for bs_2 in range(u_fea_96_2.shape[0]):                 
            video_2_st = int(transits_target[bs_2][0].item())
            video_2_ed = int(transits_target[bs_2][1].item())
            assert video_2_st < video_2_ed
            video_2_segs.append(seg_pool_1d(u_fea_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
        video_2_segs = torch.cat(video_2_segs, 0).transpose(1, 2)   

        video_1_segs_map = []
        for bs_1 in range(u_feamap_96_1.shape[0]):
            video_1_st = int(transits_data[bs_1][0].item())
            video_1_ed = int(transits_data[bs_1][1].item())
            video_1_segs_map.append(seg_pool_3d(u_feamap_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
        video_1_segs_map = torch.cat(video_1_segs_map, 0)
        video_1_segs_map = video_1_segs_map.reshape(video_1_segs_map.shape[0], video_1_segs_map.shape[1], video_1_segs_map.shape[2], -1).transpose(2, 3)
        video_1_segs_map = torch.cat([video_1_segs_map[:,:,:,i] for i in range(video_1_segs_map.shape[-1])], 2).transpose(1, 2)

        video_2_segs_map = []
        for bs_2 in range(u_fea_96_2.shape[0]):
            video_2_st = int(transits_target[bs_2][0].item())
            video_2_ed = int(transits_target[bs_2][1].item())
            video_2_segs_map.append(seg_pool_3d(u_feamap_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
        video_2_segs_map = torch.cat(video_2_segs_map, 0)
        video_2_segs_map = video_2_segs_map.reshape(video_2_segs_map.shape[0], video_2_segs_map.shape[1], video_2_segs_map.shape[2], -1).transpose(2, 3)
        video_2_segs_map = torch.cat([video_2_segs_map[:, :, :, i] for i in range(video_2_segs_map.shape[-1])], 2).transpose(1, 2)
    else:
        video_1_segs = []
        for bs_1 in range(u_fea_96_1.shape[0]):
            video_1_st = int(label_1_tas_pred[bs_1][0].item())
            video_1_ed = int(label_1_tas_pred[bs_1][1].item())
            if video_1_st == 0:
                video_1_st = 1
            if video_1_ed == 0:
                video_1_ed = 1
            video_1_segs.append(seg_pool_1d(u_fea_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
        video_1_segs = torch.cat(video_1_segs, 0).transpose(1, 2)   

        video_2_segs = []
        for bs_2 in range(u_fea_96_2.shape[0]):                 
            video_2_st = int(label_2_tas_pred[bs_2][0].item())
            video_2_ed = int(label_2_tas_pred[bs_2][1].item())
            if video_2_st == 0:
                video_2_st = 1
            if video_2_ed == 0:
                video_2_ed = 1
            video_2_segs.append(seg_pool_1d(u_fea_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
        video_2_segs = torch.cat(video_2_segs, 0).transpose(1, 2)   

        video_1_segs_map = []
        for bs_1 in range(u_feamap_96_1.shape[0]):
            video_1_st = int(label_1_tas_pred[bs_1][0].item())
            video_1_ed = int(label_1_tas_pred[bs_1][1].item())
            if video_1_st == 0:
                video_1_st = 1
            if video_1_ed == 0:
                video_1_ed = 1
            video_1_segs_map.append(seg_pool_3d(u_feamap_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
        video_1_segs_map = torch.cat(video_1_segs_map, 0)
        video_1_segs_map = video_1_segs_map.reshape(video_1_segs_map.shape[0], video_1_segs_map.shape[1], video_1_segs_map.shape[2], -1).transpose(2, 3)
        video_1_segs_map = torch.cat([video_1_segs_map[:, :, :, i] for i in range(video_1_segs_map.shape[-1])], 2).transpose(1, 2)

        video_2_segs_map = []
        for bs_2 in range(u_fea_96_2.shape[0]):
            video_2_st = int(label_2_tas_pred[bs_2][0].item())
            video_2_ed = int(label_2_tas_pred[bs_2][1].item())
            if video_2_st == 0:
                video_2_st = 1
            if video_2_ed == 0:
                video_2_ed = 1
            video_2_segs_map.append(seg_pool_3d(u_feamap_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
        video_2_segs_map = torch.cat(video_2_segs_map, 0)
        video_2_segs_map = video_2_segs_map.reshape(video_2_segs_map.shape[0], video_2_segs_map.shape[1], video_2_segs_map.shape[2], -1).transpose(2, 3)
        video_2_segs_map = torch.cat([video_2_segs_map[:, :, :, i] for i in range(video_2_segs_map.shape[-1])], 2).transpose(1, 2)

    decoder_video_12_map_list = []
    decoder_video_21_map_list = []
    
    for i in range(args.step_num):
        decoder_video_12_map = decoder(video_1_segs[:, i*args.fix_size:(i+1)*args.fix_size,:],
                                                    video_2_segs_map[:, i*args.fix_size*H*W:(i+1)*args.fix_size*H*W,:])     # N,15,256/64
        decoder_video_21_map = decoder(video_2_segs[:, i*args.fix_size:(i+1)*args.fix_size,:],
                                        video_1_segs_map[:, i*args.fix_size*H*W:(i+1)*args.fix_size*H*W,:])    # N,15,256/64
        decoder_video_12_map_list.append(decoder_video_12_map)
        decoder_video_21_map_list.append(decoder_video_21_map)

    decoder_video_12_map = torch.cat(decoder_video_12_map_list, 1)
    decoder_video_21_map = torch.cat(decoder_video_21_map_list, 1)

    return decoder_video_12_map, decoder_video_21_map

def backbone_forward_train(backbone, cls_head, ps_net, decoder, regressor, bce, mse, data, target, optimizer,
                           epoch, batch_index, batch_nums, scaler, top1_accs, pred_scores, pred_tious_5,
                           pred_tious_75, args, **kwargs):
    """Defines the computation performed at every call when training."""
    

    
    # input shape is [batch, num_clips, frames, height, width]
    if isinstance(cls_head, torch.nn.DataParallel):
        cls_head = cls_head.module
    optimizer.zero_grad()

    label, imgs_data, imgs_target, transits_data, transits_target, dive_score_data, dive_score_target = data_process_train(data, target)
    gt_label = (label - 1).squeeze()
    losses = dict()
    with autocast():
        ############# I3D featrue ###########
        cls_feamap_12, com_feature_12_u, com_feamap_12_u, N, T, C, \
        T_t, H_t, W_t = I3D_process(backbone, imgs_data, imgs_target)
        cls_score = cls_head(cls_feamap_12)
        loss_cls = cls_head.loss(cls_score, gt_label)
        losses.update(loss_cls)

        ############# Procedure Segmentation #############
        u_fea_96, transits_pred, u_feamap_96, label_12_tas, \
            loss_tas, transits_st_ed, label_1_tas_pred, label_2_tas_pred = \
                ps_net_process_train(com_feature_12_u, com_feamap_12_u, ps_net, 
                               N, H_t, W_t, transits_data, transits_target, bce)

        ############# Procedure-aware Cross-attention #############
        decoder_video_12_map, decoder_video_21_map = \
            decoder_process_train(u_fea_96, u_feamap_96, transits_data, transits_target, 
                            label_1_tas_pred, label_2_tas_pred, decoder, H_t, W_t, epoch, args)

        ############# Fine-grained Contrastive Regression #############
        decoder_12_21 = torch.cat((decoder_video_12_map, decoder_video_21_map), 0)
        delta = regressor(decoder_12_21)
        delta = delta.mean(1)


        loss_aqa = mse(delta[:delta.shape[0]//2], (dive_score_data - dive_score_target)) \
                + mse(delta[delta.shape[0]//2:], (dive_score_target - dive_score_data))

        score = (delta[:delta.shape[0]//2].detach() + dive_score_target)
        pred_scores.extend([i.item() for i in score])

        tIoU_results = []
        for bs in range(transits_pred.shape[0] // 2):
            tIoU_results.append(segment_iou(np.ar ray(label_12_tas.squeeze(-1).cpu())[bs],
                                            np.array(transits_st_ed.squeeze(-1).cpu())[bs],
                                            args))

        tiou_thresholds = np.array([0.5, 0.75])
        tIoU_correct_per_thr = cal_tiou(tIoU_results, tiou_thresholds)
        Batch_tIoU_5 = tIoU_correct_per_thr[0]
        Batch_tIoU_75 = tIoU_correct_per_thr[1]
        pred_tious_5.extend([Batch_tIoU_5])
        pred_tious_75.extend([Batch_tIoU_75])

    top1_accs.append(losses['top1_acc'].detach().cpu().numpy())
    scaler.scale(losses['loss_cls'] + loss_tas + loss_aqa).backward()
    # scaler.scale(loss_aqa).backward()
    scaler.unscale_(optimizer)
    grad_clip = getattr(args, 'grad_clip', False)
    if grad_clip:        
        for model in [backbone, ps_net, decoder, regressor]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = grad_clip.get('max_norm'), norm_type = grad_clip.get('norm_type'))

    scaler.step(optimizer)
    scaler.update()

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if batch_index % (args.print_freq) == 0:
            print('[Time: %s][Training][%d/%d][%d/%d] loss_cls: %.4f, psnet_loss: %.4f, aqa_loss: %.4f, lr1 : %0.5f, lr2 : %0.5f'
            % (end_time, epoch, args.total_epochs, batch_index, batch_nums, losses['loss_cls'].item(),
               loss_tas.item(), loss_aqa.item(), optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
              
        
def ps_net_process_test(com_feature_12_u, com_feamap_12_u, ps_net, N, H, W, transits_data, transits_target):
    u_fea_96, transits_pred = ps_net(com_feature_12_u)
    u_feamap_96, transits_pred_map = ps_net(com_feamap_12_u)
    u_feamap_96 = u_feamap_96.reshape(2 * N, u_feamap_96.shape[1], u_feamap_96.shape[2], H, W)

    label_12_tas = torch.cat((transits_data, transits_target), 0)
    # num = round(transits_pred.shape[1] / transits_pred.shape[-1])
    transits_st_ed = torch.zeros(label_12_tas.size())
    for bs in range(transits_pred.shape[0]):
        for i in range(transits_pred.shape[-1]):
            # transits_st_ed[bs, i] = transits_pred[bs, i * num: (i + 1) * num, i].argmax(0).cpu().item() + i * num
            first_transit_pred = transits_pred[bs, :-1, 0].argmax(0).cpu().item()
            transits_st_ed[bs, 0] = first_transit_pred
            second_transit_pred = transits_pred[bs, first_transit_pred + 1:, 1].argmax(0).cpu().item() + first_transit_pred + 1
            transits_st_ed[bs, 1] = second_transit_pred
    label_1_tas_pred = transits_st_ed[:transits_st_ed.shape[0] // 2]
    label_2_tas_pred = transits_st_ed[transits_st_ed.shape[0] // 2:]

    return u_fea_96, transits_pred, u_feamap_96, label_12_tas, transits_st_ed, label_1_tas_pred, label_2_tas_pred

def decoder_process_test(u_fea_96, u_feamap_96, label_1_tas_pred, label_2_tas_pred, decoder, H, W, args):
    u_fea_96_1 = u_fea_96[:u_fea_96.shape[0] // 2].transpose(1, 2)
    u_fea_96_2 = u_fea_96[u_fea_96.shape[0] // 2:].transpose(1, 2)
    u_feamap_96_1 = u_feamap_96[:u_feamap_96.shape[0] // 2].transpose(1, 2)
    u_feamap_96_2 = u_feamap_96[u_feamap_96.shape[0] // 2:].transpose(1, 2)

    video_1_segs = []
    for bs_1 in range(u_fea_96_1.shape[0]):
        video_1_st = int(label_1_tas_pred[bs_1][0].item())
        video_1_ed = int(label_1_tas_pred[bs_1][1].item())
        if video_1_st == 0:
            video_1_st = 1
        if video_1_ed == 0:
            video_1_ed = 1
        video_1_segs.append(seg_pool_1d(u_fea_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
    video_1_segs = torch.cat(video_1_segs, 0).transpose(1, 2) 

    video_2_segs = []
    for bs_2 in range(u_fea_96_2.shape[0]):                 
        video_2_st = int(label_2_tas_pred[bs_2][0].item())
        video_2_ed = int(label_2_tas_pred[bs_2][1].item())
        if video_2_st == 0:
            video_2_st = 1
        if video_2_ed == 0:
            video_2_ed = 1
        video_2_segs.append(seg_pool_1d(u_fea_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
    video_2_segs = torch.cat(video_2_segs, 0).transpose(1, 2)   

    video_1_segs_map = []
    for bs_1 in range(u_feamap_96_1.shape[0]):
        video_1_st = int(label_1_tas_pred[bs_1][0].item())
        video_1_ed = int(label_1_tas_pred[bs_1][1].item())
        if video_1_st == 0:
            video_1_st = 1
        if video_1_ed == 0:
            video_1_ed = 1
        video_1_segs_map.append(seg_pool_3d(u_feamap_96_1[bs_1].unsqueeze(0), video_1_st, video_1_ed, args.fix_size))
    video_1_segs_map = torch.cat(video_1_segs_map, 0)
    video_1_segs_map = video_1_segs_map.reshape(video_1_segs_map.shape[0], video_1_segs_map.shape[1], video_1_segs_map.shape[2], -1).transpose(2, 3)
    video_1_segs_map = torch.cat([video_1_segs_map[:, :, :, i] for i in range(video_1_segs_map.shape[-1])], 2).transpose(1, 2)

    video_2_segs_map = []
    for bs_2 in range(u_fea_96_2.shape[0]):
        video_2_st = int(label_2_tas_pred[bs_2][0].item())
        video_2_ed = int(label_2_tas_pred[bs_2][1].item())
        if video_2_st == 0:
            video_2_st = 1
        if video_2_ed == 0:
            video_2_ed = 1
        video_2_segs_map.append(seg_pool_3d(u_feamap_96_2[bs_2].unsqueeze(0), video_2_st, video_2_ed, args.fix_size))
    video_2_segs_map = torch.cat(video_2_segs_map, 0)
    video_2_segs_map = video_2_segs_map.reshape(video_2_segs_map.shape[0], video_2_segs_map.shape[1], video_2_segs_map.shape[2], -1).transpose(2, 3)
    video_2_segs_map = torch.cat([video_2_segs_map[:, :, :, i] for i in range(video_2_segs_map.shape[-1])], 2).transpose(1, 2)

    decoder_video_12_map_list = []
    decoder_video_21_map_list = []
    for i in range(args.step_num):
        decoder_video_12_map = decoder(video_1_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                                video_2_segs_map[:,
                                                i * args.fix_size * H * W:(i + 1) * args.fix_size * H * W,
                                                :])
        decoder_video_21_map = decoder(video_2_segs[:, i * args.fix_size:(i + 1) * args.fix_size, :],
                                        video_1_segs_map[:, i * args.fix_size * H * W:(i + 1) * args.fix_size * H * W,
                                        :])
        decoder_video_12_map_list.append(decoder_video_12_map)
        decoder_video_21_map_list.append(decoder_video_21_map)

    decoder_video_12_map = torch.cat(decoder_video_12_map_list, 1)
    decoder_video_21_map = torch.cat(decoder_video_21_map_list, 1)
    
    return decoder_video_12_map, decoder_video_21_map

def backbone_forward_test(backbone, cls_head, ps_net, decoder, regressor, data, targets, top1_accs, pred_scores,
                          pred_tious_test_5, pred_tious_test_75, args):
    """Defines the computation performed at every call when evaluation, testing."""

    if isinstance(cls_head, torch.nn.DataParallel):
        cls_head = cls_head.module
    score = 0
    tIoU_results = []
    imgs_data = data['imgs'].cuda()
    label_data = data['label'].cuda()
    transits_data = data['transits'].cuda()
    imgs_data = imgs_data.reshape((-1, ) + imgs_data.shape[2:])
    gt_label = (label_data - 1).reshape(-1)

    for target in targets:
        imgs_target = target['imgs'].cuda()
        transits_target = target['transits'].cuda()
        dive_score_target = target['dive_score'].cuda()

        imgs_target = imgs_target.reshape((-1, ) + imgs_target.shape[2:])
        with autocast():
            ############# I3D featrue #############
            # com_feature_12_u, com_feamap_12_u, N, C, \
            #     T, H, W = I3D_process(backbone, imgs_data, imgs_target)
            cls_feamap_12,com_feature_12_u, com_feamap_12_u, N, T, C, \
                T_t, H_t, W_t = I3D_process(backbone, imgs_data, imgs_target)
            cls_feamap_1 = cls_feamap_12[:len(imgs_data)]
            cls_score = cls_head(cls_feamap_1)
            ############# Procedure Segmentation #############
            # u_fea_96, transits_pred, u_feamap_96, label_12_tas, \
            #     transits_st_ed, label_1_tas_pred, label_2_tas_pred = \
            #         ps_net_process_test(com_feature_12_u, com_feamap_12_u, ps_net, 
            #                             N, H, W, transits_data, transits_target)
            u_fea_96, transits_pred, u_feamap_96, label_12_tas, \
                transits_st_ed, label_1_tas_pred, label_2_tas_pred = \
                    ps_net_process_test(com_feature_12_u, com_feamap_12_u, ps_net, 
                                        N, H_t, W_t, transits_data, transits_target)

            ############# Procedure-aware Cross-attention #############
            # decoder_video_12_map, decoder_video_21_map = \
            #     decoder_process_test(u_fea_96, u_feamap_96, label_1_tas_pred, 
            #                          label_2_tas_pred, decoder, H, W, args)
            decoder_video_12_map, decoder_video_21_map = \
                decoder_process_test(u_fea_96, u_feamap_96, label_1_tas_pred, 
                                     label_2_tas_pred, decoder, H_t, W_t, args)
            ############# Fine-grained Contrastive Regression #############
            decoder_12_21 = torch.cat((decoder_video_12_map, decoder_video_21_map), 0)
            delta = regressor(decoder_12_21)
            delta = delta.mean(1)

            score += (delta[:delta.shape[0]//2].detach() + dive_score_target)

            for bs in range(transits_pred.shape[0] // 2):
                tIoU_results.append(segment_iou(np.array(label_12_tas.squeeze(-1).cpu())[bs],
                                                np.array(transits_st_ed.squeeze(-1).cpu())[bs], args))

    labels = np.array(gt_label.detach().cpu().numpy())[:, np.newaxis]
    top1_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                              gt_label.detach().cpu().numpy(),
                              (1,))
    top1_accs.append(top1_acc[0])
    pred_scores.extend([i.item() / len(targets) for i in score])
    tIoU_results_mean = [sum(tIoU_results) / len(tIoU_results)]
    tiou_thresholds = np.array([0.5, 0.75])
    tIoU_correct_per_thr = cal_tiou(tIoU_results_mean, tiou_thresholds)
    pred_tious_test_5.extend([tIoU_correct_per_thr[0]])
    pred_tious_test_75.extend([tIoU_correct_per_thr[1]])

    return cls_score.detach().cpu().numpy(), labels.reshape(-1),transits_pred[:transits_pred.shape[0] // 2].detach().cpu().numpy(), transits_data.detach().cpu().numpy()


# def save_outputs(pred_scores, true_scores, transits_preds, transits_labels, args):
#     save_path_pred_scores = os.path.join(args.experiment_path, 'pred_scores.npy')
#     save_path_true_scores = os.path.join(args.experiment_path, 'true_scores.npy')
#     save_path_transits_preds = os.path.join(args.experiment_path, 'transits_preds.npy')
#     save_path_transits_labels = os.path.join(args.experiment_path, 'transits_labels.npy')
#
#     np.save(save_path_pred_scores, pred_scores)
#     np.save(save_path_true_scores, true_scores)
#     np.save(save_path_transits_preds, transits_preds)
#     np.save(save_path_transits_labels, transits_labels)

def save_checkpoint(backbone, cls_head, ps_net, decoder, regressor, optimizer, scheduler, epoch, epoch_best_aqa,
                    rho_best, L2_min, RL2_min, top1_acc, ckpt_name, args):
    torch.save({
        'backbone': backbone.state_dict(),
        'cls_head': cls_head.state_dict(),
        'ps_net': ps_net.state_dict(),
        'decoder': decoder.state_dict(),
        'regressor': regressor.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'epoch_best_aqa':epoch_best_aqa,
        'rho_best':rho_best,
        'L2_min':L2_min,
        'RL2_min':RL2_min,
        'top1_acc': top1_acc
    }, os.path.join(args.experiment_path, ckpt_name + '.pth'))




