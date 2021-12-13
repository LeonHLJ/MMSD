import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.init as torch_init
from torch.autograd import Variable
import pdb


def weights_init_random(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        # torch_init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


def calculate_l1_norm(f):
    f_norm = torch.norm(f, p=2, dim=-1, keepdim=True)
    f = torch.div(f, f_norm)
    return f


class WSTAL(nn.Module):
    def __init__(self, args):
        super().__init__()
        # feature embedding
        self.n_in = args.inp_feat_num
        self.n_out = args.out_feat_num
        self.n_class = args.class_num
        self.dropout = args.dropout

        self.rgb_stream = SingleStream(args, self.n_out)
        self.flow_stream = SingleStream(args, self.n_out)
        self.peer_stream = SingleStream(args, self.n_out)

        # rgb embedding
        self.rgb_embedding = FeatureEmbedding(self.n_in, self.n_out, self.dropout)
        # flow embedding
        self.flow_embedding = FeatureEmbedding(self.n_in, self.n_out, self.dropout)

        self.fusion_embedding = nn.Sequential(
                                    nn.Linear(self.n_in * 2, self.n_out),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(self.dropout),
                                    )

        self.apply(weights_init_random)

    def forward(self, x):
        rgb_inp = x[..., :1024]
        flow_inp = x[..., 1024:]

        r_med, r_embed = self.rgb_embedding(rgb_inp)
        f_med, f_embed = self.flow_embedding(flow_inp)
        fusion_embed = torch.cat([r_med, f_med], -1)
        fusion_embed = self.fusion_embedding(fusion_embed)

        r_ca_vid_pred, r_mil_vid_pred, r_class_agno_att, r_frm_scrs, r_norms_emb = self.rgb_stream(r_embed)
        f_ca_vid_pred, f_mil_vid_pred, f_class_agno_att, f_frm_scrs, f_norms_emb = self.flow_stream(f_embed)
        p_ca_vid_pred, p_mil_vid_pred, p_class_agno_att, p_frm_scrs, p_norms_emb = self.peer_stream(fusion_embed)

        return [r_ca_vid_pred, r_mil_vid_pred, r_class_agno_att, r_frm_scrs, r_norms_emb], \
        [f_ca_vid_pred, f_mil_vid_pred, f_class_agno_att, f_frm_scrs, f_norms_emb], \
        [p_ca_vid_pred, p_mil_vid_pred, p_class_agno_att, p_frm_scrs, p_norms_emb]


class SingleStream(nn.Module):
    def __init__(self, args, n_out):
        super().__init__()
        # read parameters
        self.n_out = n_out
        self.n_class = args.class_num
        # hyper-parameters
        self.scale_factor = args.scale_factor

        # action classifier
        self.ac_center = nn.Parameter(torch.zeros(self.n_class + 1, self.n_out))
        torch_init.xavier_uniform_(self.ac_center)
        # foreground classifier
        self.fg_center = nn.Parameter(-1.0 * self.ac_center[-1, ...][None, ...])

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_emb):
        # normalization
        norms_emb = calculate_l1_norm(x_emb)
        norms_ac = calculate_l1_norm(self.ac_center)
        norms_fg = calculate_l1_norm(self.fg_center)

        # generate class scores
        frm_scrs = torch.einsum('ntd,cd->ntc', [norms_emb, norms_ac]) * self.scale_factor
        frm_fb_scrs = torch.einsum('ntd,kd->ntk', [norms_emb, norms_fg]).squeeze(-1) * self.scale_factor

        # generate attention
        class_agno_att = self.sigmoid(frm_fb_scrs)
        class_wise_att = self.sigmoid(frm_scrs)
        class_agno_norm_att = class_agno_att / torch.sum(class_agno_att, dim=1, keepdim=True)
        class_wise_norm_att = class_wise_att / torch.sum(class_wise_att, dim=1, keepdim=True)

        # class-agnostic attention + action classification
        # foreground
        ca_vid_feat = torch.einsum('ntd,nt->nd', [x_emb, class_agno_norm_att])
        ca_vid_norm_feat = calculate_l1_norm(ca_vid_feat)
        # classification
        ca_vid_scr = torch.einsum('nd,cd->nc', [ca_vid_norm_feat, norms_ac]) * self.scale_factor
        ca_vid_pred = F.softmax(ca_vid_scr, -1)

        # MIL branch
        # temporal score aggregation
        mil_vid_feat = torch.einsum('ntd,ntc->ncd', [x_emb, class_wise_norm_att])
        mil_vid_norm_feat = calculate_l1_norm(mil_vid_feat)
        # classification
        mil_vid_scr = torch.einsum('ncd,cd->nc', [mil_vid_norm_feat, norms_ac]) * self.scale_factor
        mil_vid_pred = F.softmax(mil_vid_scr, -1)

        return ca_vid_pred, mil_vid_pred, class_agno_att, frm_scrs


class FeatureEmbedding(nn.Module):
    def __init__(self, n_in, n_out, dropout):
        super().__init__()
        self.process1 = nn.Sequential(
            nn.Linear(n_in, n_in),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.process2 = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x1 = self.process1(x)
        x2 = self.process2(x1)
        return x1, x2