import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../../"))
import torch
import numpy as np
from models.core import top_k_accuracy
from datetime import datetime
from torch.cuda.amp import autocast
import time

import torch.nn.functional as F



def backbone_forward_train(backbone, cls_head, data, target, optimizer,
                           epoch, batch_index, batch_nums, scaler, top1_accs, args):
        train_start_time = time.time()
        """Defines the computation performed at every call when training."""
        imgs_data = data['imgs'].cuda()
        label_data = data['label'].cuda()

        imgs_target = target['imgs'].cuda()
        label_target = target['label'].cuda()
        if isinstance(cls_head, torch.nn.DataParallel):
            cls_head = cls_head.module
        optimizer.zero_grad()
        imgs_data = imgs_data.reshape((-1, ) + imgs_data.shape[2:])
        imgs_target = imgs_target.reshape((-1, ) + imgs_target.shape[2:])
        label = torch.cat((label_data, label_target), 0)
        gt_label = (label-1).squeeze()
        # gt_label = (label_data-1).reshape(-1)
        losses = dict()
        with autocast():
            ############# I3D featrue ###########
            _, _, _, _, cls_feamap_12 = backbone([imgs_data, imgs_target])
            # cls_feamap_1 = cls_feamap_12[:len(imgs_data)]
            # cls_score = cls_head(cls_feamap_1)
            cls_score = cls_head(cls_feamap_12)
            loss_cls = cls_head.loss(cls_score, gt_label)
            losses.update(loss_cls)
        top1_accs.append(losses['top1_acc'].detach().cpu().numpy())
        scaler.scale(losses['loss_cls']).backward()
        scaler.unscale_(optimizer)
        grad_clip = getattr(args, 'grad_clip', False)
        if grad_clip:        
            for model in [backbone, cls_head]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = grad_clip.get('max_norm'), norm_type = grad_clip.get('norm_type'))
        scaler.step(optimizer)
        scaler.update()
        train_end_time = time.time()
        if batch_index % (args.print_freq) == 0:
              print('[Time: %s][Training][%d/%d][%d/%d], training time: %ds, loss_cls: %.4f, top1_acc: %.4f'
              % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, args.total_epochs, batch_index, batch_nums, (train_end_time-train_start_time)*args.print_freq, losses['loss_cls'].item(), losses['top1_acc'].item()))
              
def backbone_forward_test(backbone, cls_head, data, targets, top1_accs):
        """Defines the computation performed at every call when evaluation, testing."""
        
        if isinstance(cls_head, torch.nn.DataParallel):
            cls_head = cls_head.module
        imgs_data = data['imgs'].cuda()
        label_data = data['label'].cuda()
        imgs_data = imgs_data.reshape((-1, ) + imgs_data.shape[2:])
        gt_label = (label_data-1).reshape(-1)

        target = targets[-1]
        imgs_target = target['imgs'].cuda()
        imgs_target = imgs_target.reshape((-1, ) + imgs_target.shape[2:])
        with autocast():
            ############# I3D featrue #############
            _, _, _, _, cls_feamap_12 = backbone([imgs_data, imgs_target])
            cls_feamap_1 = cls_feamap_12[:len(imgs_data)]
            cls_score = cls_head(cls_feamap_1)
        max_1_preds = np.argsort(cls_score.detach().cpu().numpy(), axis=1)[:, -1:][:, ::-1]
        labels = np.array(gt_label.detach().cpu().numpy())[:, np.newaxis]
        top1_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       gt_label.detach().cpu().numpy(), 
                                       (1, ))
        top1_accs.append(top1_acc[0])
        return cls_score.detach().cpu().numpy(), labels.reshape(-1)


def backbone_forward_train_single(backbone, cls_head, data, optimizer,
                           epoch, batch_index, batch_nums, scaler, top1_accs, args):
        train_start_time = time.time()
        """Defines the computation performed at every call when training."""
        imgs_data = data['imgs'].cuda()
        label_data = data['label'].cuda()

        if isinstance(cls_head, torch.nn.DataParallel):
            cls_head = cls_head.module
        optimizer.zero_grad()
        imgs_data = imgs_data.reshape((-1, ) + imgs_data.shape[2:])
        gt_label = (label_data-1).reshape(-1)
        losses = dict()
        with autocast():
            ############# I3D featrue ###########
            feamap = backbone(imgs_data)
            cls_score = cls_head(feamap)
            loss_cls = cls_head.loss(cls_score, gt_label)
            losses.update(loss_cls)
        top1_accs.append(losses['top1_acc'].detach().cpu().numpy())
        scaler.scale(losses['loss_cls']).backward()
        scaler.unscale_(optimizer)
        grad_clip = getattr(args, 'grad_clip', False)
        if grad_clip:
            for model in [backbone, cls_head]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = grad_clip.get('max_norm'), norm_type = grad_clip.get('norm_type'))
        
        scaler.step(optimizer)
        scaler.update()
        train_end_time = time.time()
        if batch_index % (args.print_freq) == 0:
              print('[Time: %s][Training][%d/%d][%d/%d], training time: %ds, loss_cls: %.4f, top1_acc: %.4f'
              % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, args.total_epochs, batch_index, batch_nums, (train_end_time-train_start_time)*args.print_freq, losses['loss_cls'].item(), losses['top1_acc'].item()))
              
def backbone_forward_test_single(backbone, cls_head, data, top1_accs):
        """Defines the computation performed at every call when evaluation, testing."""
        
        if isinstance(cls_head, torch.nn.DataParallel):
            cls_head = cls_head.module
        imgs_data = data['imgs'].cuda()
        label_data = data['label'].cuda()
        imgs_data = imgs_data.reshape((-1, ) + imgs_data.shape[2:])
        gt_label = (label_data-1).reshape(-1)
        with autocast():

            ############# I3D featrue #############
            feamap = backbone(imgs_data)
            cls_score = cls_head(feamap)
        max_1_preds = np.argsort(cls_score.detach().cpu().numpy(), axis=1)[:, -1:][:, ::-1]
        labels = np.array(gt_label.detach().cpu().numpy())[:, np.newaxis]
        top1_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       gt_label.detach().cpu().numpy(), 
                                       (1, ))
        top1_accs.append(top1_acc[0])
        return cls_score.detach().cpu().numpy(), labels.reshape(-1)

def save_cls_outputs(preds, gt_labels, args):
    save_path_pred = os.path.join(args.experiment_path, 'pred_cls.npy')
    save_path_true = os.path.join(args.experiment_path, 'true_cls.npy')
    np.save(save_path_pred, preds)
    np.save(save_path_true, gt_labels)

def save_cls_checkpoint(backbone, cls_head, optimizer, scheduler, epoch, top1_acc, ckpt_name, args):
    torch.save({
        'backbone': backbone.state_dict(),
        'cls_head': cls_head.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'top1_acc': top1_acc
    }, os.path.join(args.experiment_path, ckpt_name + '.pth'))


# def average_clip(cls_score):
#     assert len(cls_score.shape) == 3  # * (Batch, NumSegs, Dim)
#     average_clips = 'prob'
#     if average_clips == 'prob':
#         return F.softmax(cls_score, dim=2).mean(dim=1)
