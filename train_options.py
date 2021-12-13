import argparse

parser = argparse.ArgumentParser(description='WSTAL')

# basic setting
parser.add_argument('--gpus', type=int, default=[0], nargs='+', help='used gpu')
parser.add_argument('--run-type', type=int, default=0, help='train (0) or evaluate (1)')
parser.add_argument('--model-id', type=int, default=1, help='model id for saving model')

# loading model
parser.add_argument('--pretrained', action='store_true', help='is pretrained model')
parser.add_argument('--load-epoch', type=int, default=None, help='epoch of loaded model')

# storing parameters
parser.add_argument('--save-interval', type=int, default=10, help='interval for storing model')

# dataset patameters
parser.add_argument('--dataset-root', default='path_to_your_dataset', help='dataset root path')
parser.add_argument('--dataset-name', default='Thumos14reduced', help='dataset to train on')

# model settings
parser.add_argument('--feature-type', type=str, default='I3D', help='type of feature to be used (default: I3D)')
parser.add_argument('--inp-feat-num', type=int, default=1024, help='size of input feature (default: 1024)')
parser.add_argument('--out-feat-num', type=int, default=1024, help='size of output feature (default: 1024)')
parser.add_argument('--class-num', type=int, default=20, help='number of classes (default: )')
parser.add_argument('--scale-factor', type=float, default=20.0, help='scale factor for cosine similarity')

# training paramaters
parser.add_argument('--batch-size', type=int, default=10, help='number of instances in a batch of data (default: 10)')
parser.add_argument('--optimizer', type=str, default='Adam', help='used optimizer')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate (default: 0.00005)')
parser.add_argument('--weight-decay', type=float, default=0.001, help='weight deacy (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
parser.add_argument('--dropout', default=0.6, help='dropout value (default: 0.6)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--max-epoch', type=int, default=240, help='maximum epoch to train (default: 240)')

parser.add_argument('--alpha', default=0.9, help='hyper-parameter')
parser.add_argument('--propotion', default=8.0, help='hyper-parameter')

parser.add_argument('--cls_hyp', default=1.0, help='hyper-parameter')
parser.add_argument('--att_hyp', default=0.1, help='hyper-parameter')
parser.add_argument('--pkd_hyp', default=0.1, help='hyper-parameter')
parser.add_argument('--pdl_hyp', default=2.0, help='hyper-parameter')

parser.add_argument('--con_hyp', default=0.6, help='hyper-parameter')
parser.add_argument('--int_hyp', default=0.3, help='hyper-parameter')

parser.add_argument('--theta', default=0.55, help='hyper-parameter')
parser.add_argument('--T', default=0.2, help='hyper-parameter')

parser.add_argument('--lambda-caa', default=1.0, help='balancing hyper-parameter of class-agnostic attention branch')
parser.add_argument('--lambda-csa', default=0.5, help='balancing hyper-parameter of class-specific attention branch')

parser.add_argument('--iter_list', type=int, default=[40, 80, 120, 160, 200, 240], help='recalculate pseudo labels step')

# testing paramaters
parser.add_argument('--class-threshold', type=float, default=0.2, help='class threshold for rejection')
parser.add_argument('--start-threshold', type=float, default=0.0003, help='start threshold for action localization')
parser.add_argument('--end-threshold', type=float, default=0.01, help='end threshold for action localization')
parser.add_argument('--threshold-interval', type=float, default=0.0005, help='threshold interval for action localization')

# Learning Rate Decay
parser.add_argument('--decay-type', type=int, default=1, help='weight decay type (0 for None, 1 for step decay, 2 for cosine decay)')
parser.add_argument('--changeLR_list', type=int, default=[70, 1000], help='change lr step')
