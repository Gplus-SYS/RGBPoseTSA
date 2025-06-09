import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../../"))

import torch
import numpy as np
from datetime import datetime
from torch.cuda.amp import autocast
from utils.misc import segment_iou, cal_tiou

def data_process_train(data, target):
    imgs_data = data['imgs'].cuda()
    transits_data = data['transits'].cuda()
    dive_score_data = data['dive_score'].cuda()

    imgs_target = target['imgs'].cuda()
    transits_target = target['transits'].cuda()
    dive_score_target = target['dive_score'].cuda()
    
    # input shape is [batch, num_clips, frames, height, width]
    
    imgs_data = imgs_data.reshape((-1, ) + imgs_data.shape[2:])
    imgs_target = imgs_target.reshape((-1, ) + imgs_target.shape[2:])

    return imgs_data, imgs_target, transits_data, transits_target, dive_score_data, dive_score_target

def I3D_process(backbone, imgs_data, imgs_target):
    feature_1, feature_2, feamap_1, feamap_2, _ = backbone([imgs_data, imgs_target])
    N, T, C, T_t, H_t, W_t = feamap_1.size()
    feamap_1 = feamap_1.mean(-3)
    feamap_2 = feamap_2.mean(-3)
    feamap_1_re = feamap_1.reshape(-1, T, C)
    feamap_2_re = feamap_2.reshape(-1, T, C)
    com_feature_12_u = torch.cat((feature_1, feature_2), 0)
    com_feamap_12_u = torch.cat((feamap_1_re, feamap_2_re), 0)
    return com_feature_12_u, com_feamap_12_u, N, T, C, T_t, H_t, W_t


    

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
        # for i in range(transits_pred.shape[-1]):
        #     transits_st_ed[bs, i] = transits_pred[bs, i * num: (i + 1) * num, i].argmax(0).cpu().item() + i * num
        first_transit_pred = transits_pred[bs, :-1, 0].argmax(0).cpu().item()
        transits_st_ed[bs, 0] = first_transit_pred
        second_transit_pred = transits_pred[bs, first_transit_pred + 1:, 1].argmax(0).cpu().item() + first_transit_pred + 1
        transits_st_ed[bs, 1] = second_transit_pred

    label_1_tas_pred = transits_st_ed[:transits_st_ed.shape[0] // 2]
    label_2_tas_pred = transits_st_ed[transits_st_ed.shape[0] // 2:]

    return u_fea_96, transits_pred, u_feamap_96, label_12_tas, loss_tas, transits_st_ed, label_1_tas_pred, label_2_tas_pred


def backbone_forward_train(backbone, ps_net, bce, data, target, optimizer,
                           epoch, batch_index, batch_nums, scaler, pred_tious_5,
                           pred_tious_75, args, **kwargs):
    """Defines the computation performed at every call when training."""
    

    
    # input shape is [batch, num_clips, frames, height, width]
    optimizer.zero_grad()

    imgs_data, imgs_target, transits_data, transits_target, dive_score_data, dive_score_target = data_process_train(data, target)

    with autocast():
        ############# I3D featrue ###########
        com_feature_12_u, com_feamap_12_u, N, T, C, \
        T_t, H_t, W_t = I3D_process(backbone, imgs_data, imgs_target)
        ############# Procedure Segmentation #############
        u_fea_96, transits_pred, u_feamap_96, label_12_tas, \
            loss_tas, transits_st_ed, label_1_tas_pred, label_2_tas_pred = \
                ps_net_process_train(com_feature_12_u, com_feamap_12_u, ps_net, 
                               N, H_t, W_t, transits_data, transits_target, bce)

        tIoU_results = []
        for bs in range(transits_pred.shape[0] // 2):
            tIoU_results.append(segment_iou(np.array(label_12_tas.squeeze(-1).cpu())[bs],
                                            np.array(transits_st_ed.squeeze(-1).cpu())[bs],
                                            args))

        tiou_thresholds = np.array([0.5, 0.75])
        tIoU_correct_per_thr = cal_tiou(tIoU_results, tiou_thresholds)
        Batch_tIoU_5 = tIoU_correct_per_thr[0]
        Batch_tIoU_75 = tIoU_correct_per_thr[1]
        pred_tious_5.extend([Batch_tIoU_5])
        pred_tious_75.extend([Batch_tIoU_75])

    scaler.scale(loss_tas).backward()
    scaler.unscale_(optimizer)
    grad_clip = getattr(args, 'grad_clip', False)
    if grad_clip:        
        for model in [backbone, ps_net]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = grad_clip.get('max_norm'), norm_type = grad_clip.get('norm_type'))

    scaler.step(optimizer)
    scaler.update()

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if batch_index % (args.print_freq) == 0:
            print('[Time: %s][Training][%d/%d][%d/%d] psnet_loss: %.4f, lr1 : %0.5f, lr2 : %0.5f'
            % (end_time, epoch, args.total_epochs, batch_index, batch_nums, loss_tas.item(),
               optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))


def backbone_forward_train_single(backbone, ps_net, bce, data, optimizer,
                           epoch, batch_index, batch_nums, scaler, pred_tious_5,
                           pred_tious_75, args):
    """Defines the computation performed at every call when training."""

    optimizer.zero_grad()

    imgs_data = data['imgs'].cuda()
    transits_data = data['transits'].cuda()
    # input shape is [batch, num_clips, frames, height, width]
    imgs_data = imgs_data.reshape((-1,) + imgs_data.shape[2:])

    with autocast():
        ############# I3D featrue ###########
        feature, _, _ = backbone(imgs_data)
        ############# Procedure Segmentation #############
        _, transits_pred = ps_net(feature)
        label_pad = torch.zeros(transits_pred.size())
        for bs in range(transits_pred.shape[0]):
            label_pad[bs, int(transits_data[bs, 0]), 0] = 1
            label_pad[bs, int(transits_data[bs, -1]), -1] = 1
        loss_tas = bce(transits_pred, label_pad.cuda())

        # num = round(transits_pred.shape[1] / transits_pred.shape[-1])
        transits_st_ed = torch.zeros(transits_data.size())
        for bs in range(transits_pred.shape[0]):
            # for i in range(transits_pred.shape[-1]):
            #     transits_st_ed[bs, i] = transits_pred[bs, i * num: (i + 1) * num, i].argmax(0).cpu().item() + i * num
            first_transit_pred = transits_pred[bs, :-1, 0].argmax(0).cpu().item()
            transits_st_ed[bs, 0] = first_transit_pred
            second_transit_pred = transits_pred[bs, first_transit_pred + 1:, 1].argmax(
                0).cpu().item() + first_transit_pred + 1
            transits_st_ed[bs, 1] = second_transit_pred

        tIoU_results = []
        for bs in range(transits_pred.shape[0]):
            tIoU_results.append(segment_iou(np.array(transits_data.squeeze(-1).cpu())[bs],
                                            np.array(transits_st_ed.squeeze(-1).cpu())[bs],
                                            args))

        tiou_thresholds = np.array([0.5, 0.75])
        tIoU_correct_per_thr = cal_tiou(tIoU_results, tiou_thresholds)
        Batch_tIoU_5 = tIoU_correct_per_thr[0]
        Batch_tIoU_75 = tIoU_correct_per_thr[1]
        pred_tious_5.extend([Batch_tIoU_5])
        pred_tious_75.extend([Batch_tIoU_75])

    scaler.scale(loss_tas).backward()
    scaler.unscale_(optimizer)
    grad_clip = getattr(args, 'grad_clip', False)
    if grad_clip:
        for model in [backbone, ps_net]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip.get('max_norm'),
                                           norm_type=grad_clip.get('norm_type'))
    scaler.step(optimizer)
    scaler.update()

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if batch_index % (args.print_freq) == 0:
        print('[Time: %s][Training][%d/%d][%d/%d] psnet_loss: %.4f, lr1 : %0.5f, lr2 : %0.5f'
              % (end_time, epoch, args.total_epochs, batch_index, batch_nums, loss_tas.item(),
                 optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']))
              
        
def ps_net_process_test(com_feature_12_u, com_feamap_12_u, ps_net, N, H, W, transits_data, transits_target):
    u_fea_96, transits_pred = ps_net(com_feature_12_u)
    u_feamap_96, transits_pred_map = ps_net(com_feamap_12_u)
    u_feamap_96 = u_feamap_96.reshape(2 * N, u_feamap_96.shape[1], u_feamap_96.shape[2], H, W)

    label_12_tas = torch.cat((transits_data, transits_target), 0)
    # num = round(transits_pred.shape[1] / transits_pred.shape[-1])
    transits_st_ed = torch.zeros(label_12_tas.size())
    for bs in range(transits_pred.shape[0]):
        # for i in range(transits_pred.shape[-1]):
        #     transits_st_ed[bs, i] = transits_pred[bs, i * num: (i + 1) * num, i].argmax(0).cpu().item() + i * num
        first_transit_pred = transits_pred[bs, :-1, 0].argmax(0).cpu().item()
        transits_st_ed[bs, 0] = first_transit_pred
        second_transit_pred = transits_pred[bs, first_transit_pred + 1:, 1].argmax(0).cpu().item() + first_transit_pred + 1
        transits_st_ed[bs, 1] = second_transit_pred
    label_1_tas_pred = transits_st_ed[:transits_st_ed.shape[0] // 2]
    label_2_tas_pred = transits_st_ed[transits_st_ed.shape[0] // 2:]

    return u_fea_96, transits_pred, u_feamap_96, label_12_tas, transits_st_ed, label_1_tas_pred, label_2_tas_pred


def backbone_forward_test(backbone, ps_net, data, targets, pred_tious_test_5, pred_tious_test_75, args):
    """Defines the computation performed at every call when evaluation, testing."""

    tIoU_results = []
    imgs_data = data['imgs'].cuda()
    transits_data = data['transits'].cuda()
    imgs_data = imgs_data.reshape((-1, ) + imgs_data.shape[2:])

    for target in targets:
        imgs_target = target['imgs'].cuda()
        transits_target = target['transits'].cuda()

        imgs_target = imgs_target.reshape((-1, ) + imgs_target.shape[2:])
        with autocast():
            ############# I3D featrue #############
            com_feature_12_u, com_feamap_12_u, N, T, C, \
                T_t, H_t, W_t = I3D_process(backbone, imgs_data, imgs_target)
            ############# Procedure Segmentation #############
            u_fea_96, transits_pred, u_feamap_96, label_12_tas, \
                transits_st_ed, label_1_tas_pred, label_2_tas_pred = \
                    ps_net_process_test(com_feature_12_u, com_feamap_12_u, ps_net, 
                                        N, H_t, W_t, transits_data, transits_target)

            for bs in range(transits_pred.shape[0] // 2):
                tIoU_results.append(segment_iou(np.array(label_12_tas.squeeze(-1).cpu())[bs],
                                                np.array(transits_st_ed.squeeze(-1).cpu())[bs], args))

    tIoU_results_mean = [sum(tIoU_results) / len(tIoU_results)]
    tiou_thresholds = np.array([0.5, 0.75])
    tIoU_correct_per_thr = cal_tiou(tIoU_results_mean, tiou_thresholds)
    pred_tious_test_5.extend([tIoU_correct_per_thr[0]])
    pred_tious_test_75.extend([tIoU_correct_per_thr[1]])

    return transits_pred[:transits_pred.shape[0] // 2].detach().cpu().numpy(), transits_data.detach().cpu().numpy()

def backbone_forward_test_single(backbone, ps_net, data, pred_tious_test_5, pred_tious_test_75, args):
    """Defines the computation performed at every call when evaluation, testing."""

    tIoU_results = []
    imgs_data = data['imgs'].cuda()
    transits_data = data['transits'].cuda()
    imgs_data = imgs_data.reshape((-1, ) + imgs_data.shape[2:])

    with autocast():
        ############# I3D featrue ###########
        feature, _, _ = backbone(imgs_data)
        ############# Procedure Segmentation #############
        _, transits_pred = ps_net(feature)

        # num = round(transits_pred.shape[1] / transits_pred.shape[-1])
        transits_st_ed = torch.zeros(transits_data.size())

        for bs in range(transits_pred.shape[0]):
            # for i in range(transits_pred.shape[-1]):
            #     transits_st_ed[bs, i] = transits_pred[bs, i * num: (i + 1) * num, i].argmax(0).cpu().item() + i * num
            first_transit_pred = transits_pred[bs, :-1, 0].argmax(0).cpu().item()
            transits_st_ed[bs, 0] = first_transit_pred
            second_transit_pred = transits_pred[bs, first_transit_pred + 1:, 1].argmax(0).cpu().item() + first_transit_pred + 1
            transits_st_ed[bs, 1] = second_transit_pred

        for bs in range(transits_pred.shape[0]):
            tIoU_results.append(segment_iou(np.array(transits_data.squeeze(-1).cpu())[bs],
                                            np.array(transits_st_ed.squeeze(-1).cpu())[bs], args))

    tiou_thresholds = np.array([0.5, 0.75])
    tIoU_correct_per_thr = cal_tiou(tIoU_results, tiou_thresholds)
    pred_tious_test_5.extend([tIoU_correct_per_thr[0]])
    pred_tious_test_75.extend([tIoU_correct_per_thr[1]])

    return transits_pred.detach().cpu().numpy(), transits_data.detach().cpu().numpy()


def save_outputs(transits_preds, transits_labels, args):
    save_path_transits_preds = os.path.join(args.experiment_path, 'transits_preds.npy')
    save_path_transits_labels = os.path.join(args.experiment_path, 'transits_labels.npy')

    np.save(save_path_transits_preds, transits_preds)
    np.save(save_path_transits_labels, transits_labels)

def save_checkpoint(backbone, ps_net, optimizer, scheduler, epoch, pred_tious_best_5, pred_tious_best_75, ckpt_name, args):
    torch.save({
        'backbone': backbone.state_dict(),
        'ps_net': ps_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'pred_tious_best_5':pred_tious_best_5,
        'pred_tious_best_75':pred_tious_best_75,
    }, os.path.join(args.experiment_path, ckpt_name + '.pth'))




