import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import utils.utils as utils
from .classificationMAP import getClassificationMAP as cmAP
from .detectionMAP import getDetectionMAP as dtmAP


def generate_pseudo(dataloader, model, args, device):
    idxs = []
    rgb_pseudo_labels = []
    flow_pseudo_labels = []
    peer_pseudo_labels = []
    rgb_conf_scrs = []
    flow_conf_scrs = []
    peer_conf_scrs = []

    for num, sample in enumerate(dataloader):
        if (num + 1) % 100 == 0:
            print('Processing train data %d of %d' % (num + 1, len(dataloader)))

        idx = sample['idx']
        features = sample['data'].numpy()
        labels = sample['labels'].numpy()
        rgb_plbl = sample['rgb_plbl'].numpy()
        flow_plbl = sample['flow_plbl'].numpy()
        peer_plbl = sample['peer_plbl'].numpy()
        rgb_lscr = sample['rgb_lscr'].numpy()
        flow_lscr = sample['flow_lscr'].numpy()
        peer_lscr = sample['peer_lscr'].numpy()

        features = torch.from_numpy(features).float().to(device)
        labels = torch.from_numpy(labels).float().to(device)
        ab_labels = torch.cat([labels, torch.ones(labels.size(0), 1).to(device)], -1)
        
        with torch.no_grad():
            rgb_out, flow_out, peer_out = model(Variable(features))

            rgb_att = rgb_out[2]
            flow_att = flow_out[2]
            peer_att = peer_out[2]

            rgb_pred = F.softmax(rgb_out[3], -1)
            flow_pred = F.softmax(flow_out[3], -1)
            peer_pred = F.softmax(peer_out[3], -1)
            rgb_pred = rgb_pred * ab_labels[:, None, :]
            flow_pred = flow_pred * ab_labels[:, None, :]
            peer_pred = peer_pred * ab_labels[:, None, :]
            rgb_conf_scr = torch.abs(rgb_pred[..., :-1].sum(-1) - rgb_pred[..., -1])
            flow_conf_scr = torch.abs(flow_pred[..., :-1].sum(-1) - flow_pred[..., -1])
            peer_conf_scr = torch.abs(peer_pred[..., :-1].sum(-1) - peer_pred[..., -1])

            rgb_att = np.squeeze(rgb_att.cpu().data.numpy(), axis=0)
            flow_att = np.squeeze(flow_att.cpu().data.numpy(), axis=0)
            peer_att = np.squeeze(peer_att.cpu().data.numpy(), axis=0)

            rgb_conf_scr = np.squeeze(rgb_conf_scr.cpu().data.numpy(), axis=0)
            flow_conf_scr = np.squeeze(flow_conf_scr.cpu().data.numpy(), axis=0)
            peer_conf_scr = np.squeeze(peer_conf_scr.cpu().data.numpy(), axis=0)

            rgb_plbl = np.squeeze(rgb_plbl, axis=0)
            flow_plbl = np.squeeze(flow_plbl, axis=0)
            peer_plbl = np.squeeze(peer_plbl, axis=0)

            rgb_lscr = np.squeeze(rgb_lscr, axis=0)
            flow_lscr = np.squeeze(flow_lscr, axis=0)
            peer_lscr = np.squeeze(peer_lscr, axis=0)

            rgb_pseudo_label = (rgb_att > args.theta) * 1.0
            flow_pseudo_label = (flow_att > args.theta) * 1.0
            peer_pseudo_label = (peer_att > args.theta) * 1.0

            rgb_conf_mask = (rgb_conf_scr > rgb_lscr) * 1.0
            flow_conf_mask = (flow_conf_scr > flow_lscr) * 1.0
            peer_conf_mask = (peer_conf_scr > peer_lscr) * 1.0
            
            rgb_pseudo_label = rgb_conf_mask * rgb_pseudo_label + (1 - rgb_conf_mask) * rgb_plbl
            flow_pseudo_label = flow_conf_mask * flow_pseudo_label + (1 - flow_conf_mask) * flow_plbl
            peer_pseudo_label = peer_conf_mask * peer_pseudo_label + (1 - peer_conf_mask) * peer_plbl
            rgb_conf_scr = rgb_conf_mask * rgb_conf_scr + (1 - rgb_conf_mask) * rgb_lscr
            flow_conf_scr = flow_conf_mask * flow_conf_scr + (1 - flow_conf_mask) * flow_lscr
            peer_conf_scr = peer_conf_mask * peer_conf_scr + (1 - peer_conf_mask) * peer_lscr

        rgb_pseudo_labels.append(rgb_pseudo_label)
        flow_pseudo_labels.append(flow_pseudo_label)
        peer_pseudo_labels.append(peer_pseudo_label)
        rgb_conf_scrs.append(rgb_conf_scr)
        flow_conf_scrs.append(flow_conf_scr)
        peer_conf_scrs.append(peer_conf_scr)
        idxs.append(idx)

    rgb_pseudo_labels = np.array(rgb_pseudo_labels)
    flow_pseudo_labels = np.array(flow_pseudo_labels)
    peer_pseudo_labels = np.array(peer_pseudo_labels)
    rgb_conf_scrs = np.array(rgb_conf_scrs)
    flow_conf_scrs = np.array(flow_conf_scrs)
    peer_conf_scrs = np.array(peer_conf_scrs)

    return [rgb_pseudo_labels, flow_pseudo_labels, peer_pseudo_labels, rgb_conf_scrs, flow_conf_scrs, peer_conf_scrs], idxs

def ss_eval(epoch, dataloader, args, logger, model, device):
    vid_preds = []
    frm_preds = []
    vid_lens = []
    labels = []

    for num, sample in enumerate(dataloader):
        if (num + 1) % 100 == 0:
            print('Testing test data point %d of %d' % (num + 1, len(dataloader)))

        features = sample['data'].numpy()
        label = sample['labels'].numpy()
        vid_len = sample['vid_len'].numpy()

        features = torch.from_numpy(features).float().to(device)

        with torch.no_grad():
            rgb_out, flow_out, peer_out = model(Variable(features))

            vid_att = 0.2 * rgb_out[2] + 0.4 * flow_out[2] + 0.4 * peer_out[2]
            vid_pred = 0.2 * rgb_out[0] + 0.4 * flow_out[0] + 0.4 * peer_out[0]
            frm_pred = 0.2 * F.softmax(rgb_out[3], -1) + 0.4 * F.softmax(flow_out[3], -1) + 0.4 * F.softmax(peer_out[3], -1)

            frm_pred = frm_pred * vid_att[..., None]
            vid_pred = np.squeeze(vid_pred.cpu().data.numpy(), axis=0)
            frm_pred = np.squeeze(frm_pred.cpu().data.numpy(), axis=0)
            label = np.squeeze(label, axis=0)

        vid_preds.append(vid_pred)
        frm_preds.append(frm_pred)
        vid_lens.append(vid_len)
        labels.append(label)

    vid_preds = np.array(vid_preds)
    frm_preds = np.array(frm_preds)
    vid_lens = np.array(vid_lens)
    labels = np.array(labels)

    cmap = cmAP(vid_preds, labels)
    dmap, iou = dtmAP(vid_preds, frm_preds, vid_lens, dataloader.dataset.path_to_annotations, args)

    sum = 0
    count = 0
    print('Classification map %f' % cmap)
    for item in list(zip(iou, dmap)):
        print('Detection map @ %f = %f' % (item[0], item[1]))
        sum = sum + item[1]
        count += 1

    logger.log_value('Test Classification mAP', cmap, epoch)
    for item in list(zip(dmap, iou)):
        logger.log_value('Test Detection1 mAP @ IoU = ' + str(item[1]), item[0], epoch)

    print('average map = %f' % (sum / count))

    utils.write_results_to_file(args, dmap, sum / count, cmap, epoch)