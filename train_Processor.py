from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from evaluation.eval import ss_eval, generate_pseudo
from model.main_branch import WSTAL
from model.losses import NormalizedCrossEntropy, FrmScrLoss, AttLoss, CategoryCrossEntropy
from utils.video_dataloader import VideoDataset
from tensorboard_logger import Logger


class Processor():
    def __init__(self, args):
        # parameters
        self.args = args
        # create logger
        log_dir = './logs/' + self.args.dataset_name + '/' + str(self.args.model_id)
        self.logger = Logger(log_dir)
        # device
        self.device = torch.device(
            'cuda:' + str(self.args.gpu_ids[0]) if torch.cuda.is_available() and len(self.args.gpu_ids) > 0 else 'cpu')

        # dataloader
        if self.args.dataset_name in ['Thumos14', 'Thumos14reduced']:
            if self.args.run_type == 0:
                self.train_dataset = VideoDataset(self.args, 'train')
                self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                                     batch_size=1,
                                                                     shuffle=True,
                                                                     num_workers=2 * len(self.args.gpu_ids),
                                                                     drop_last=False)
                self.test_data_loader = torch.utils.data.DataLoader(VideoDataset(self.args, 'test'), batch_size=1,
                                                                    shuffle=False, drop_last=False)
            elif self.args.run_type == 1:
                self.test_data_loader = torch.utils.data.DataLoader(VideoDataset(self.args, 'test'), batch_size=1,
                                                                    shuffle=False, drop_last=False)
        else:
            raise ValueError('Do Not Exist This Dataset')

        # Loss Function Setting
        self.loss_nce = NormalizedCrossEntropy()
        self.loss_att = FrmScrLoss(self.args.propotion)
        self.loss_pkd = CategoryCrossEntropy(self.args.T)
        self.loss_pd = nn.MSELoss(reduction='none')

        # Model Setting
        self.model = WSTAL(self.args).to(self.device)

        # Model Parallel Setting
        if len(self.args.gpu_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            self.model_module = self.model.module
        else:
            self.model_module = self.model

        # Loading Pretrained Model
        if self.args.pretrained:
            model_dir = './ckpt/' + self.args.dataset_name + '/' + str(self.args.model_id) + '/' + str(
                self.args.load_epoch) + '.pkl'
            if os.path.isfile(model_dir):
                self.model_module.load_state_dict(torch.load(model_dir))
            else:
                raise ValueError('Do Not Exist This Pretrained File')

        # Optimizer Setting
        if self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=[0.9, 0.99],
                                              weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                             weight_decay=self.args.weight_decay, nesterov=True)
        else:
            raise ValueError('Do Not Exist This Optimizer')

        # Optimizer Parallel Setting
        if len(self.args.gpu_ids) > 1:
            self.optimizer = nn.DataParallel(self.optimizer, device_ids=self.args.gpu_ids)
            self.optimizer_module = self.optimizer.module
        else:
            self.optimizer_module = self.optimizer

    def processing(self):
        if self.args.run_type == 0:
            self.train()
        elif self.args.run_type == 1:
            self.val(self.args.load_epoch)
        else:
            raise ValueError('Do not Exist This Processing')

    def train(self):
        print('Start training!')
        self.model_module.train(mode=True)
        if self.args.pretrained:
            epoch_range = range(self.args.load_epoch, self.args.max_epoch)
        else:
            epoch_range = range(self.args.max_epoch)

        iter = 0
        step = 0
        current_lr = self.args.lr
        loss_recorder = {
            'cls': 0,
            'att': 0,
            'pkd': 0,
            'pdl': 0,
        }
        for epoch in epoch_range:
            for num, sample in enumerate(self.train_data_loader):
                if self.args.decay_type == 0:
                    for param_group in self.optimizer_module.param_groups:
                        param_group['lr'] = current_lr
                elif self.args.decay_type == 1:
                    if num == 0:
                        current_lr = self.Step_decay_lr(epoch)
                        for param_group in self.optimizer_module.param_groups:
                            param_group['lr'] = current_lr
                elif self.args.decay_type == 2:
                    current_lr = self.Cosine_decay_lr(epoch, num)
                    for param_group in self.optimizer_module.param_groups:
                        param_group['lr'] = current_lr

                iter = iter + 1
                features = sample['data'].numpy()
                labels = sample['labels'].numpy()
                rgb_plbl = sample['rgb_plbl'].numpy()
                flow_plbl = sample['flow_plbl'].numpy()
                peer_plbl = sample['peer_plbl'].numpy()

                labels = torch.from_numpy(labels).float().to(self.device)
                features = torch.from_numpy(features).float().to(self.device)
                rgb_plbl = torch.from_numpy(rgb_plbl).float().to(self.device)
                flow_plbl = torch.from_numpy(flow_plbl).float().to(self.device)
                peer_plbl = torch.from_numpy(peer_plbl).float().to(self.device)

                ab_labels = torch.cat([labels, torch.ones(labels.size(0), 1).to(self.device)], -1)
                awb_labels = torch.cat([labels, torch.zeros(labels.size(0), 1).to(self.device)], -1)

                rgb_out, flow_out, peer_out = self.model(features)

                rgb_cls_loss = self.loss_nce(rgb_out[0], awb_labels) * self.args.lambda_caa \
                              + self.loss_nce(rgb_out[1], ab_labels) * self.args.lambda_csa
                flow_cls_loss = self.loss_nce(flow_out[0], awb_labels) * self.args.lambda_caa \
                              + self.loss_nce(flow_out[1], ab_labels) * self.args.lambda_csa
                peer_cls_loss = self.loss_nce(peer_out[0], awb_labels) * self.args.lambda_caa \
                              + self.loss_nce(peer_out[1], ab_labels) * self.args.lambda_csa

                # attention regularization
                rgb_att_loss = self.loss_att(F.sigmoid(rgb_out[3]), ab_labels)
                flow_att_loss = self.loss_att(F.sigmoid(flow_out[3]), ab_labels)
                peer_att_loss = self.loss_att(F.sigmoid(peer_out[3]), ab_labels)

                # knowledge distillation
                pkd_r2f_loss = self.loss_pkd(rgb_out[3], flow_out[3])
                pkd_f2r_loss = self.loss_pkd(flow_out[3], rgb_out[3])
                pkd_rf2p_loss = self.loss_pkd(peer_out[3], rgb_out[3] / 2 + flow_out[3] / 2)

                cls_loss = rgb_cls_loss + flow_cls_loss + peer_cls_loss
                att_loss = rgb_att_loss + flow_att_loss + peer_att_loss
                pkd_loss = pkd_r2f_loss + pkd_f2r_loss + pkd_rf2p_loss

                total_loss = cls_loss * self.args.cls_hyp + att_loss * self.args.att_hyp + pkd_loss * self.args.pkd_hyp

                loss_recorder['cls'] += cls_loss.item()
                loss_recorder['att'] += att_loss.item()
                loss_recorder['pkd'] += pkd_loss.item()

                if epoch >= self.args.iter_list[0]:
                    rgb_pred = F.softmax(rgb_out[3], -1)
                    flow_pred = F.softmax(flow_out[3], -1)
                    peer_pred = F.softmax(peer_out[3], -1)
                    reliable_out = self.generate_reliable_label(rgb_pred, flow_pred, peer_pred,\
                     rgb_plbl, flow_plbl, peer_plbl, ab_labels)

                    # pseudo label loss
                    rgb_gd_loss = self.loss_pd(rgb_out[2], reliable_out[0])
                    flow_gd_loss = self.loss_pd(flow_out[2], reliable_out[1])
                    peer_gd_loss = self.loss_pd(peer_out[2], reliable_out[2])
                    rgb_gd_loss = torch.masked_select(rgb_gd_loss, reliable_out[3])
                    flow_gd_loss = torch.masked_select(flow_gd_loss, reliable_out[4])
                    peer_gd_loss = torch.masked_select(peer_gd_loss, reliable_out[5])

                    rgb_gd_loss = rgb_gd_loss.mean(-1).mean(-1) if len(rgb_gd_loss) > 0 else torch.tensor(0).to(self.device)
                    flow_gd_loss = flow_gd_loss.mean(-1).mean(-1) if len(flow_gd_loss) > 0 else torch.tensor(0).to(self.device)
                    peer_gd_loss = peer_gd_loss.mean(-1).mean(-1) if len(peer_gd_loss) > 0 else torch.tensor(0).to(self.device)

                    pdl_loss = rgb_gd_loss + flow_gd_loss + peer_gd_loss
                    total_loss += pdl_loss * self.args.pdl_hyp
                    loss_recorder['pdl'] += pdl_loss.item()

                total_loss.backward()

                if iter % self.args.batch_size == 0:
                    step += 1
                    print('Epoch: {}/{}, Iter: {:02d}, Lr: {:.6f}'.format(
                        epoch + 1,
                        self.args.max_epoch,
                        step,
                        current_lr), end=' ')
                    for k, v in loss_recorder.items():
                        print('Loss_{}: {:.4f}'.format(k, v / self.args.batch_size), end=' ')
                        loss_recorder[k] = 0

                    print()
                    self.optimizer_module.step()
                    self.optimizer_module.zero_grad()

            if (epoch + 1) in self.args.iter_list:
                self.model_module.eval()
                pseudo_out, idxs = generate_pseudo(self.train_data_loader, self.model_module, self.args, self.device)
                self.model_module.train()
                self.train_dataset.assign_pseudo_gt(pseudo_out, idxs)
                self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                                     batch_size=1,
                                                                     shuffle=True,
                                                                     num_workers=2 * len(self.args.gpu_ids),
                                                                     drop_last=False)

            if (epoch + 1) % self.args.save_interval == 0:
                out_dir = './ckpt/' + self.args.dataset_name + '/' + str(self.args.model_id) + '/' + str(
                    epoch + 1) + '.pkl'
                torch.save(self.model_module.state_dict(), out_dir)
                self.model_module.eval()
                ss_eval(epoch + 1, self.test_data_loader, self.args, self.logger, self.model_module, self.device)
                self.model_module.train()

    def val(self, epoch):
        print('Start testing!')
        self.model_module.eval()
        ss_eval(epoch, self.test_data_loader, self.args, self.logger, self.model_module, self.device)
        print('Finish testing!')

    def generate_reliable_label(self, rgb_pred, flow_pred, peer_pred, rgb_plbl, flow_plbl, peer_plbl, label):
        rgb_pred = rgb_pred * label[:, None, :]
        flow_pred = flow_pred * label[:, None, :]
        peer_pred = peer_pred * label[:, None, :]
        rgb_conf_scr = torch.abs(rgb_pred[..., :-1].sum(-1) - rgb_pred[..., -1])
        flow_conf_scr = torch.abs(flow_pred[..., :-1].sum(-1) - flow_pred[..., -1])
        peer_conf_scr = torch.abs(peer_pred[..., :-1].sum(-1) - peer_pred[..., -1])

        # single stream stream
        rgb_single_mask = rgb_conf_scr.ge(self.args.con_hyp)
        flow_single_mask = flow_conf_scr.ge(self.args.con_hyp)
        peer_single_mask = peer_conf_scr.ge(self.args.con_hyp)

        # rgb stream
        fmr_mask = (flow_conf_scr - rgb_conf_scr).ge(self.args.int_hyp)
        fmr_mask = torch.logical_and(fmr_mask, flow_single_mask)

        pmr_mask = (peer_conf_scr - rgb_conf_scr).ge(self.args.int_hyp)
        pmr_mask = torch.logical_and(pmr_mask, peer_single_mask)

        rgb_final_mask = torch.logical_or(fmr_mask, rgb_single_mask)
        rgb_final_mask = torch.logical_or(pmr_mask, rgb_final_mask)
        rgb_final_lbl = (rgb_plbl * rgb_single_mask + flow_plbl * fmr_mask + peer_plbl * pmr_mask) \
                        / (rgb_single_mask * 1.0 + fmr_mask * 1.0 + pmr_mask * 1.0 + 1e-4)

        # flow stream
        rmf_mask = (rgb_conf_scr - flow_conf_scr).ge(self.args.int_hyp)
        rmf_mask = torch.logical_and(rmf_mask, rgb_single_mask)

        pmf_mask = (peer_conf_scr - flow_conf_scr).ge(self.args.int_hyp)
        pmf_mask = torch.logical_and(pmf_mask, peer_single_mask)

        flow_final_mask = torch.logical_or(rmf_mask, flow_single_mask)
        flow_final_mask = torch.logical_or(pmf_mask, flow_final_mask)
        flow_final_lbl = (flow_plbl * flow_single_mask + rgb_plbl * rmf_mask + peer_plbl * pmf_mask) \
                        / (flow_single_mask * 1.0 + rmf_mask * 1.0 + pmf_mask * 1.0 + 1e-4)

        # peer stream
        rmp_mask = (rgb_conf_scr - peer_conf_scr).ge(self.args.int_hyp)
        rmp_mask = torch.logical_and(rmp_mask, rgb_single_mask)

        fmp_mask = (flow_conf_scr - peer_conf_scr).ge(self.args.int_hyp)
        fmp_mask = torch.logical_and(fmp_mask, flow_single_mask)

        peer_final_mask = torch.logical_or(rmp_mask, peer_single_mask)
        peer_final_mask = torch.logical_or(fmp_mask, peer_final_mask)

        peer_final_lbl = (peer_plbl * peer_single_mask + rgb_plbl * rmp_mask + flow_plbl * fmp_mask) \
                        / (peer_single_mask * 1.0 + rmp_mask * 1.0 + fmp_mask * 1.0 + 1e-4)

        rgb_final_lbl = Variable(rgb_final_lbl.detach().data, requires_grad=False)
        flow_final_lbl = Variable(flow_final_lbl.detach().data, requires_grad=False)
        peer_final_lbl = Variable(peer_final_lbl.detach().data, requires_grad=False)
        rgb_final_mask = Variable(rgb_final_mask.detach().data, requires_grad=False)
        flow_final_mask = Variable(flow_final_mask.detach().data, requires_grad=False)
        peer_final_mask = Variable(peer_final_mask.detach().data, requires_grad=False)

        return [rgb_final_lbl, flow_final_lbl, peer_final_lbl, rgb_final_mask, flow_final_mask, peer_final_mask]


    def Step_decay_lr(self, epoch):
        lr_list = []
        current_epoch = epoch + 1
        for i in range(0, len(self.args.changeLR_list) + 1):
            lr_list.append(self.args.lr * (0.2 ** i))

        lr_range = self.args.changeLR_list.copy()
        lr_range.insert(0, 0)
        lr_range.append(self.args.max_epoch + 1)

        if len(self.args.changeLR_list) != 0:
            for i in range(0, len(lr_range) - 1):
                if lr_range[i + 1] >= current_epoch > lr_range[i]:
                    lr_step = i
                    break

        current_lr = lr_list[lr_step]
        return current_lr

    def Cosine_decay_lr(self, epoch, batch):
        if self.args.warmup:
            max_epoch = self.args.max_epoch - self.args.warmup_epoch
            current_epoch = epoch + 1 - self.args.warmup_epoch
        else:
            max_epoch = self.args.max_epoch
            current_epoch = epoch + 1

        current_lr = 1 / 2.0 * (1.0 + np.cos(
            (current_epoch * self.args.batch_num + batch) / (max_epoch * self.args.batch_num) * np.pi)) * self.args.lr

        return current_lr
