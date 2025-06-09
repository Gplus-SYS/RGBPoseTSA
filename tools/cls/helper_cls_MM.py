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

              
def MM_backbone_forward_train(rgb_backbone, pose_backbone, cls_head, data, target,
                              optimizer, epoch, batch_index, batch_nums, scaler, top1_accs,args):
        train_start_time = time.time()
        """Defines the computation performed at every call when training."""
        imgs_data = data['imgs'].cuda()
        heatmap_imgs_data = data['heatmap_imgs'].cuda()
        label_data = data['label'].cuda()
        
        imgs_target = target['imgs'].cuda()
        heatmap_imgs_target = target['heatmap_imgs'].cuda()
        label_target = target['label'].cuda()
        if isinstance(cls_head, torch.nn.DataParallel):
            cls_head = cls_head.module
        optimizer.zero_grad()
        imgs_data = imgs_data.reshape((-1, ) + imgs_data.shape[2:])
        heatmap_imgs_data = heatmap_imgs_data.reshape((-1, ) + heatmap_imgs_data.shape[2:])
        imgs_target = imgs_target.reshape((-1, ) + imgs_target.shape[2:])
        heatmap_imgs_target = heatmap_imgs_target.reshape((-1, ) + heatmap_imgs_target.shape[2:])
        label = torch.cat((label_data, label_target), 0)
        gt_label = (label-1).reshape(-1)
        losses = dict()
        with autocast():
            _, _, _, _, cls_feamap_rgb_12 = rgb_backbone(imgs_data, imgs_target)
            _, _, _, _, cls_feamap_pose_12 =  pose_backbone(heatmap_imgs_data, heatmap_imgs_target)
            x = torch.cat((cls_feamap_rgb_12, cls_feamap_pose_12), dim=1)
            cls_score = cls_head(x)
            loss_cls = cls_head.loss(cls_score, gt_label)
            losses.update(loss_cls)
        top1_accs.append(losses['top1_acc'].detach().cpu().numpy())
        scaler.scale(losses['loss_cls']).backward()
        scaler.unscale_(optimizer)
        grad_clip = getattr(args, 'grad_clip', False)
        if grad_clip:        
            for model in [rgb_backbone, pose_backbone, cls_head]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = grad_clip.get('max_norm'), norm_type = grad_clip.get('norm_type'))
        scaler.step(optimizer)
        scaler.update()
        train_end_time = time.time()
        if batch_index % (args.print_freq) == 0:
              print('[Time: %s][Training][%d/%d][%d/%d], training time: %ds, loss_cls: %.4f, top1_acc: %.4f'
              % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),epoch, args.total_epochs, batch_index, batch_nums, (train_end_time-train_start_time)*args.print_freq, losses['loss_cls'].item(), losses['top1_acc'].item()))

        return losses

def MM_backbone_forward_test(rgb_backbone, pose_backbone, cls_head, data, targets, top1_accs):
        """Defines the computation performed at every call when evaluation, testing."""

        if isinstance(cls_head, torch.nn.DataParallel):
            cls_head = cls_head.module
        imgs_data = data['imgs'].cuda()
        heatmap_imgs_data = data['heatmap_imgs'].cuda()
        label_data = data['label'].cuda()
        imgs_data = imgs_data.reshape((-1, ) + imgs_data.shape[2:])
        heatmap_imgs_data = heatmap_imgs_data.reshape((-1, ) + heatmap_imgs_data.shape[2:])
        gt_label = (label_data-1).reshape(-1)
        target = targets[-1]
        imgs_target = target['imgs'].cuda()
        heatmap_imgs_target = target['heatmap_imgs'].cuda()
        imgs_target = imgs_target.reshape((-1, ) + imgs_target.shape[2:])
        heatmap_imgs_target = heatmap_imgs_target.reshape((-1, ) + heatmap_imgs_target.shape[2:])
        with autocast():
            ############# I3D featrue #############
            _, _, _, _, cls_feamap_rgb_12 = rgb_backbone(imgs_data, imgs_target)
            _, _, _, _, cls_feamap_pose_12 =  pose_backbone(heatmap_imgs_data, heatmap_imgs_target)
            cls_feamap_com_12 = torch.cat((cls_feamap_rgb_12, cls_feamap_pose_12), dim=1)
            cls_feamap_com_1 = cls_feamap_com_12[:len(imgs_target)]
            cls_score = cls_head(cls_feamap_com_1)
        max_1_preds = np.argsort(cls_score.detach().cpu().numpy(), axis=1)[:, -1:][:, ::-1]
        labels = np.array(gt_label)[:, np.newaxis]
        top1_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       gt_label.detach().cpu().numpy(), 
                                       (1, ))
        top1_accs.append(top1_acc[0])
        return cls_score.detach().cpu().numpy(), labels.reshape(-1)

def MM_backbone_forward_train_single(rgb_backbone, pose_backbone, cls_head, data, optimizer, 
                                     epoch, batch_index, batch_nums, scaler, top1_accs,args):
        train_start_time = time.time()
        """Defines the computation performed at every call when training."""
        imgs_data = data['imgs'].cuda()
        heatmap_imgs_data = data['heatmap_imgs'].cuda()
        label_data = data['label'].cuda()

        if isinstance(cls_head, torch.nn.DataParallel):
            cls_head = cls_head.module
        optimizer.zero_grad()
        imgs_data = imgs_data.reshape((-1, ) + imgs_data.shape[2:])
        heatmap_imgs_data = heatmap_imgs_data.reshape((-1, ) + heatmap_imgs_data.shape[2:])
        
        gt_label = (label_data-1).reshape(-1)
        losses = dict()
        with autocast():
            feamap_rgb = rgb_backbone(imgs_data)
            feamap_pose = pose_backbone(heatmap_imgs_data)
            x = torch.cat((feamap_rgb, feamap_pose), dim=1)
            cls_score = cls_head(x)
            loss_cls = cls_head.loss(cls_score, gt_label)
            losses.update(loss_cls)
        top1_accs.append(losses['top1_acc'].detach().cpu().numpy())
        scaler.scale(losses['loss_cls']).backward()
        scaler.unscale_(optimizer)
        grad_clip = args.optimizer_config.grad_clip
        if grad_clip.get('enable'):
            for model in [rgb_backbone, pose_backbone, cls_head]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = grad_clip.get('max_norm'), norm_type = grad_clip.get('norm_type'))
        scaler.step(optimizer)
        scaler.update()
        train_end_time = time.time()
        if batch_index % (args.print_freq) == 0:
              print('[Time: %s][Training][%d/%d][%d/%d], training time: %ds, loss_cls: %.4f, top1_acc: %.4f'
              % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),epoch, args.total_epochs, batch_index, batch_nums, (train_end_time-train_start_time)*args.print_freq, losses['loss_cls'].item(), losses['top1_acc'].item()))

        return losses

def MM_backbone_forward_test_single(rgb_backbone, pose_backbone, cls_head, data, top1_accs):
        """Defines the computation performed at every call when evaluation, testing."""

        if isinstance(cls_head, torch.nn.DataParallel):
            cls_head = cls_head.module
        imgs_data = data['imgs'].cuda()
        heatmap_imgs_data = data['heatmap_imgs'].cuda()
        label_data = data['label'].cuda()
        imgs_data = imgs_data.reshape((-1, ) + imgs_data.shape[2:])
        heatmap_imgs_data = heatmap_imgs_data.reshape((-1, ) + heatmap_imgs_data.shape[2:])
        gt_label = (label_data-1).reshape(-1)
        
        with autocast():
            ############# I3D featrue #############
            feamap_rgb = rgb_backbone(imgs_data)
            feamap_pose = pose_backbone(heatmap_imgs_data)
            x = torch.cat((feamap_rgb, feamap_pose), dim=1)
            cls_score = cls_head(x)
        max_1_preds = np.argsort(cls_score.detach().cpu().numpy(), axis=1)[:, -1:][:, ::-1]
        labels = np.array(gt_label)[:, np.newaxis]
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


def MM_save_cls_checkpoint(rgb_backbone, pose_backbone, cls_head, optimizer, scheduler, epoch, top1_acc, ckpt_name, args):
    torch.save({
        'rgb_backbone': rgb_backbone.state_dict(),
        'pose_backbone': pose_backbone.state_dict(),
        'cls_head': cls_head.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'top1_acc': top1_acc,
    }, os.path.join(args.experiment_path, ckpt_name + '.pth'))

