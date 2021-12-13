import torch
import numpy as np
import utils.utils as utils
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, args, run_type):
        self.args = args
        self.run_type = run_type
        self.dataset_name = args.dataset_name
        self.path_to_features = args.dataset_root + '%s/%s-%s-JOINTFeatures.npy' % (
            args.dataset_name, args.dataset_name, args.feature_type)
        self.path_to_annotations = args.dataset_root + self.dataset_name + '/'
        self.features = np.load(self.path_to_features, encoding='bytes')
        self.labels = np.load(self.path_to_annotations + 'labels_all.npy')
        self.classlist = np.load(self.path_to_annotations + 'classlist.npy')
        self.subset = np.load(self.path_to_annotations + 'subset.npy')
        self.rgb_pseudo_labels = [0 for _ in range(len(self.features))]
        self.flow_pseudo_labels = [0 for _ in range(len(self.features))]
        self.peer_pseudo_labels = [0 for _ in range(len(self.features))]
        self.rgb_lbl_scrs = [0 for _ in range(len(self.features))]
        self.flow_lbl_scrs = [0 for _ in range(len(self.features))]
        self.peer_lbl_scrs = [0 for _ in range(len(self.features))]
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0
        self.labels_multihot = [utils.strlist2multihot(labs, self.classlist) for labs in self.labels]
        self.train_test_idx()
        self.classwise_feature_mapping()

    def __len__(self):
        if self.run_type == 'train':
            return int(len(self.trainidx))
        else:
            return int(len(self.testidx))

    def assign_pseudo_gt(self, pseudo_out, idxs):
        self.rgb_pseudo_labels = [pseudo_out[0][idxs.index(i)] if i in idxs else 0 for i in range(len(self.features))]
        self.flow_pseudo_labels = [pseudo_out[1][idxs.index(i)] if i in idxs else 0 for i in range(len(self.features))]
        self.peer_pseudo_labels = [pseudo_out[2][idxs.index(i)] if i in idxs else 0 for i in range(len(self.features))]
        self.rgb_lbl_scrs = [pseudo_out[3][idxs.index(i)] if i in idxs else 0 for i in range(len(self.features))]
        self.flow_lbl_scrs = [pseudo_out[4][idxs.index(i)] if i in idxs else 0 for i in range(len(self.features))]
        self.peer_lbl_scrs = [pseudo_out[5][idxs.index(i)] if i in idxs else 0 for i in range(len(self.features))]

    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s.decode('utf-8') == 'validation':  # Specific to Thumos14
                self.trainidx.append(i)
            else:
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category.decode('utf-8'):
                        idx.append(i)
                        break
            self.classwiseidx.append(idx)

    def __getitem__(self, idx):
        sample = dict()
        if self.run_type == 'train':
            labs = self.labels_multihot[self.trainidx[idx]]
            feat = self.features[self.trainidx[idx]]
            rgb_plbl = self.rgb_pseudo_labels[self.trainidx[idx]]
            flow_plbl = self.flow_pseudo_labels[self.trainidx[idx]]
            peer_plbl = self.peer_pseudo_labels[self.trainidx[idx]]
            rgb_lscr = self.rgb_lbl_scrs[self.trainidx[idx]]
            flow_lscr = self.flow_lbl_scrs[self.trainidx[idx]]
            peer_lscr = self.peer_lbl_scrs[self.trainidx[idx]]

            sample['data'] = feat
            sample['labels'] = labs
            sample['rgb_plbl'] = rgb_plbl
            sample['flow_plbl'] = flow_plbl
            sample['peer_plbl'] = peer_plbl
            sample['rgb_lscr'] = rgb_lscr
            sample['flow_lscr'] = flow_lscr
            sample['peer_lscr'] = peer_lscr
            sample['idx'] = idx
        elif self.run_type == 'test':
            labs = self.labels_multihot[self.testidx[idx]]
            feat = self.features[self.testidx[idx]]
            sample['vid_len'] = feat.shape[0]
            sample['data'] = feat
            sample['labels'] = labs
        return sample
