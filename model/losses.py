import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class NormalizedCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, labels):
        new_labels = labels / (torch.sum(labels, dim=1, keepdim=True) + 1e-8)
        loss = -1.0 * torch.mean(torch.sum(Variable(new_labels) * torch.log(pred), dim=1), dim=0)
        return loss

class CategoryCrossEntropy(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, pred, soft_label):
        soft_label = F.softmax(soft_label / self.T, -1)
        soft_label = Variable(soft_label.detach().data, requires_grad=False)
        loss = -1.0 * torch.sum(Variable(soft_label) * torch.log_softmax(pred / self.T, -1), dim=-1)
        loss = loss.mean(-1).mean(-1)
        return loss

class FrmScrLoss(nn.Module):
    def __init__(self, propotion):
        super().__init__()
        self.s = propotion

    def forward(self, frm_scrs, label):
        n, t, c = frm_scrs.size()

        # class-wise attention contrast loss
        max_frm_values, _ = torch.topk(frm_scrs, max(int(t // self.s), 1), 1)
        mean_max_frm = max_frm_values.mean(1)

        min_frm_values, _ = torch.topk(-frm_scrs, max(int(t // self.s), 1), 1)
        mean_min_frm = -min_frm_values.mean(1)

        temporal_loss = (mean_min_frm - mean_max_frm) * label
        temporal_loss = temporal_loss.sum(-1).mean(-1)

        # foreground-background separation
        frm_scrs = frm_scrs * label[:, None, :]
        frm_act_scrs = frm_scrs[..., :-1]
        frm_bck_scr = frm_scrs[..., -1]
        
        frm_act_scr = frm_act_scrs.max(-1)[0]
        categorcial_loss = -1.0 * torch.abs(frm_act_scr - frm_bck_scr).mean(-1).mean(-1)

        return temporal_loss + categorcial_loss