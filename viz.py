#import mayavi.mlab as mlab
import argparse
import glob
from pathlib import Path
import open3d as o3d
from utils import visualize_o3d as V_o3d

import numpy as np
import torch
import os
import json


#from tools.visual_utils import visualize_utils as V
#from tools.visual_utils import visualize_o3d as V_o3d

import pandas as pd

EXCEL_PATH = "~/Documents/votenet/rio/config/Classes.xlsx"
df = pd.read_excel(EXCEL_PATH, sheet_name='Mapping')

def label2idx(label):
    # input: global label : "sofa"
    # output: index in rio7: 1
    return df[df['Label'] == label]['RIO7 Index'].iloc[0]

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pointnet2.pytorch_utils import BNMomentumScheduler
from utils.tf_visualizer import Visualizer as TfVisualizer
from models.ap_helper import APCalculator, parse_predictions, parse_groundtruths

writer = SummaryWriter("log/loss")

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='votenet', help='Model file name [default: votenet]')
parser.add_argument('--dataset', default='3rscan', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresh', type=float, default=0.25, help='AP IoU threshold [default: 0.25]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='80,120,160', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.5,0.4,0.3', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use V2 box labels for SUN RGB-D dataset')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--dump_results', action='store_true', help='Dump results.')
FLAGS = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
LOG_DIR = FLAGS.log_dir
DEFAULT_DUMP_DIR = os.path.join(BASE_DIR, os.path.basename(LOG_DIR))
DUMP_DIR = FLAGS.dump_dir if FLAGS.dump_dir is not None else DEFAULT_DUMP_DIR
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH
FLAGS.DUMP_DIR = DUMP_DIR

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and FLAGS.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)'%(LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s %s'%(LOG_DIR, DUMP_DIR))

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create Dataset and Dataloader
"""
if FLAGS.dataset == 'sunrgbd':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, MAX_NUM_OBJ
    from model_util_sunrgbd import SunrgbdDatasetConfig
    DATASET_CONFIG = SunrgbdDatasetConfig()
    TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=NUM_POINT,
        augment=True,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
        use_v1=(not FLAGS.use_sunrgbd_v2))
    TEST_DATASET = SunrgbdDetectionVotesDataset('val', num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height),
        use_v1=(not FLAGS.use_sunrgbd_v2))
elif FLAGS.dataset == 'scannet':
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset import ScannetDetectionDataset, MAX_NUM_OBJ
    from model_util_scannet import ScannetDatasetConfig
    DATASET_CONFIG = ScannetDatasetConfig()
    TRAIN_DATASET = ScannetDetectionDataset('train', num_points=NUM_POINT,
        augment=True,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
    TEST_DATASET = ScannetDetectionDataset('val', num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
"""
if FLAGS.dataset == '3rscan':
    sys.path.append(os.path.join(ROOT_DIR, 'rio'))
    from rio import RIO, MAX_NUM_OBJ

    TRAIN_DATASET = RIO(True, '/home/vink/Documents/votenet/rio/data/', 20000, False)
    TEST_DATASET = RIO(False, '/home/vink/Documents/votenet/rio/data/', 20000, False)

else:
    print('Unknown dataset %s. Exiting...'%(FLAGS.dataset))
    exit(-1)

print(len(TRAIN_DATASET), len(TEST_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, batch_size=1,
    shuffle=False, num_workers=4, worker_init_fn=my_worker_init_fn, drop_last=True)
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=1,
    shuffle=False, num_workers=4, worker_init_fn=my_worker_init_fn, drop_last=True)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))

# Init the model and optimzier
# MODEL = importlib.import_module(FLAGS.model) # import network module
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# num_input_channel = int(FLAGS.use_color)*3 + int(not FLAGS.no_height)*1


MEAN_SIZE_PATH = os.getcwd() + "/rio/config/mean_size.npy"
MEAN_SIZE = np.load(MEAN_SIZE_PATH)

eval_config_dict = {'remove_empty_box': False, 'use_3d_nms': True, 'nms_iou': 0.01,
        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
        'conf_thresh': 0.3}
#
# if FLAGS.model == 'boxnet':
#     Detector = MODEL.BoxNet
# else:
#     Detector = MODEL.VoteNet
MODEL = importlib.import_module(FLAGS.model) # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if FLAGS.model == 'boxnet':
    Detector = MODEL.BoxNet
else:
    Detector = MODEL.VoteNet

num_input_channel = 1 # only height here
net = Detector(num_class=8,
               num_heading_bin=12,
               num_size_cluster=8,
               mean_size_arr=MEAN_SIZE,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling)

net.load_state_dict(torch.load('pretrained/199'))
net.to(device)
#net = torch.load('pretrained/archive/data.pkl')
net.eval()

def main():

    with torch.no_grad():
        for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].to(device)
            inputs = {'point_clouds': batch_data_label['point_clouds']}
            #data_dict = batch_data_label['scan_dict']
            end_points = net(inputs)
            #end_points['point_clouds'] = inputs['point_clouds']

            for key in batch_data_label:
                assert (key not in end_points)
                end_points[key] = batch_data_label[key]

            pred_map_cls,pred_tracking_out,pred_cls_out = parse_predictions(end_points, eval_config_dict)  #after nms
            _,gt_out = parse_groundtruths(end_points,None)

            data_dict = {
                'gt_boxes': gt_out
            }
            pred_dicts = {
                'pred_boxes':pred_tracking_out,
                'pred_labels':pred_cls_out
            }

            geometry_list = []
            #geometry_list = V_o3d.get_boxes(data_dict['gt_boxes'][:,:,:7],False,geometry_list,data_dict['gt_boxes'][:,:,7])
            geometry_list = V_o3d.get_boxes(pred_dicts['pred_boxes'],True,geometry_list,pred_dicts['pred_labels'])
            geometry_list = V_o3d.get_pcd_from_np(inputs['point_clouds'][:,1:],geometry_list)


            # o3d.visualization.RenderOption.point_size = 1.0
            # o3d.visualization.RenderOption.line_width = 6.0
            o3d.visualization.draw_geometries(geometry_list)#,point_size=1.0,line_width=2.0)



if __name__ == '__main__':
    main()
