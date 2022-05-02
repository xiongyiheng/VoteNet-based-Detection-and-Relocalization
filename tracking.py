#import mayavi.mlab as mlab
import argparse
import glob
from pathlib import Path
import open3d as o3d


import numpy as np
import torch
import os
import json


#from tools.visual_utils import visualize_utils as V
#from tools.visual_utils import visualize_o3d as V_o3d

import pandas as pd

EXCEL_PATH = "~/Documents/votenet/dataset/config/Classes.xlsx"
df = pd.read_excel(EXCEL_PATH, sheet_name='Mapping')

def label2idx(label):
    # input: global label : "sofa"
    # output: index in rio7: 1
    return df[df['Label'] == label]['RIO7 Index'].iloc[0]

import json

def extract_scans_info(data_path):
    # output: write the scans and reference info to a .json file.
    # usage: only run once

    # xyz offset of "scans" compared to "reference"
    #       if it is "reference" -> xyz offset = 0

    with open(data_path+"3RScan.json",'r') as load_f:
        load_dict = json.load(load_f)

        for i in range(len(load_dict)):
            ref_id = load_dict[i]["reference"]
            scan_list = []

            if load_dict[i]["type"] != "test":
                for j in range(len(load_dict[i]["scans"])):
                    scan_id = load_dict[i]["scans"][j]["reference"]
                    obj_movement = load_dict[i]["scans"][j]["rigid"]  #"rigid" in 3RScan json file
                    cam_trans = load_dict[i]["scans"][j]["transform"]
                    scan_list.append(scan_id)
                    dict_scan={
                        "ref_id": ref_id,
                        "obj_movement": obj_movement,
                        "is_ref": False,
                        "cam_trans":cam_trans
                    }
                    with open(data_path+scan_id+'/scans.json','w') as write_scan:
                        write_scan.write(json.dumps(dict_scan,ensure_ascii = False))
                    write_scan.close()

                    dict_ref = {
                        "ref_id": ref_id,
                        "is_ref": True,
                        "is_test": False,
                        "scan_list": scan_list, # all scans in this ref scene
                        "obj_id": None
                    }
            else:
                for k in range(len(load_dict[i]["scans"])):
                    scan_id = load_dict[i]["scans"][k]["reference"]
                    scan_list.append(scan_id)
                    obj_movement = obj_movement = load_dict[i]["scans"][k]["rigid"] # for test rescans, no .json file
                dict_ref = {
                    "ref_id": ref_id,
                    "is_ref": True,
                    "is_test": True,
                    "scan_list": scan_list,  # all scans in this ref scene
                    "obj_id": obj_movement
                }

            with open(data_path + ref_id + '/scans.json', 'w') as write_f:
                write_f.write(json.dumps(dict_ref, ensure_ascii=False))
            write_f.close()
            print(ref_id)

def write_scan_list(data_path):

    with open(data_path + "3RScan.json", 'r') as load_f:
        load_dict = json.load(load_f)

        scan_list_eval = []
        scan_list_test = []

        for i in range(len(load_dict)):
            ref_id = load_dict[i]["reference"]


            if load_dict[i]["type"] != "test":
                for j in range(len(load_dict[i]["scans"])):
                    scan_id = load_dict[i]["scans"][j]["reference"]+"\n"
                    obj_movement = load_dict[i]["scans"][j]["rigid"]  # "rigid" in 3RScan json file
                    cam_trans = load_dict[i]["scans"][j]["transform"]
                    scan_list_eval.append(scan_id)

                    # with open(data_path + scan_id + '/scans.json', 'w') as write_scan:
                    #     write_scan.write(json.dumps(dict_scan, ensure_ascii=False))
                    # write_scan.close()


            else:
                for k in range(len(load_dict[i]["scans"])):
                    scan_id = load_dict[i]["scans"][k]["reference"]+"\n"
                    scan_list_test.append(scan_id)
                    obj_movement = obj_movement = load_dict[i]["scans"][k]["rigid"]  # for test rescans, no .json file

            with open(data_path + '/scans_eval.txt', 'w') as write_eval:
                write_eval.writelines(scan_list_eval)
                write_eval.close()
            with open(data_path + '/scans_test.txt', 'w') as write_test:
                write_test.writelines(scan_list_test)
                write_test.close()

            print(ref_id)



DATA_PATH = '/home/vink/Documents/votenet/dataset/3RScan/data/'
extract_scans_info(DATA_PATH)
write_scan_list(DATA_PATH)

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

    TRAIN_DATASET = RIO(True, '/media/jingsong/5ebd8121-a4eb-43b3-b486-532a6238f6cf/', 20000, True)
    TEST_DATASET = RIO(False, '/media/jingsong/5ebd8121-a4eb-43b3-b486-532a6238f6cf/', 20000, False)

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
num_input_channel = 1 # only height here

MEAN_SIZE_PATH = os.getcwd() + "/rio/config/mean_size.npy"
MEAN_SIZE = np.load(MEAN_SIZE_PATH)

eval_config_dict = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
        'conf_thresh': 0.5}
#
# if FLAGS.model == 'boxnet':
#     Detector = MODEL.BoxNet
# else:
#     Detector = MODEL.VoteNet
MODEL = importlib.import_module('votenet') # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Detector = MODEL.VoteNet
net = Detector(num_class=8,
               num_heading_bin=12,
               num_size_cluster=8,
               mean_size_arr=MEAN_SIZE,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling)

net = torch.load('pretrained/archive/data.pkl')
# net = torch.load('modelpath.pt')
net.to(device)
net.eval()

def main():

    aliagn = False

    TP_sum = 0
    FN_sum = 0
    FP_sum = 0
    t_sum = 0.0
    a_sum = 0.0
    recall = 0.0
    precision = 0.0
    t_median = []
    a_median = []
    with torch.no_grad():
        for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
            for key in batch_data_label:
                batch_data_label[key] = batch_data_label[key].to(device)
            inputs = {'point_clouds': batch_data_label['point_clouds']}
            data_dict = batch_data_label['scan_dict']
            end_points = net(inputs)
            end_points['point_clouds'] = inputs['point_clouds']
            pred_map_cls,pred_tracking_out,pred_cls_out = parse_predictions(end_points, eval_config_dict)  #after nms

            pred_dicts=[]
            pred_map = {'pred_boxes':pred_tracking_out,
                        'pred_labels':pred_cls_out}

            pred_dicts.append(pred_map)

            # aliagn boxes in rescan to ref
            cam_matrix, inquire_id_ls,trans_ls,ref_id = extract_info(data_dict)

            if aliagn:
                pred_boxes = aliagn_box(pred_dicts[0]['pred_boxes'], cam_matrix)
            else:
                pred_boxes = pred_dicts[0]['pred_boxes'][0,:,:]

            #if pred_dicts[0]['pred_labels'] is not None and not isinstance(pred_dicts[0]['pred_labels'], np.ndarray):
            pred_labels = pred_dicts[0]['pred_labels'][0,:,:]  #(N,7)

            # extract center and size of inquire id in ref
            gt_boxes_inquire = extract_gt_boxes(ref_id, inquire_id_ls)

            # assign target_boxes to each inquire
            TP, FN, FP, t,angle,t_ls,a_ls = assign_target_eval(gt_boxes_inquire,pred_boxes,pred_labels,cam_matrix,trans_ls)

            TP_sum += TP
            FN_sum += FN
            FP_sum += FP
            t_sum += t
            a_sum += angle
            t_median.extend(t_ls)
            a_median.extend(a_ls)


            # # VIZ
            # geometry_list = []
            # geometry_list = V_o3d.get_boxes(data_dict['gt_boxes'][0,:,:7],False,geometry_list)
            # geometry_list = V_o3d.get_boxes(pred_dicts[0]['pred_boxes'],True,geometry_list)
            # geometry_list = V_o3d.get_pcd_from_np(data_dict['points'][:,1:],geometry_list)
            #
            #
            #
            # o3d.visualization.draw_geometries(geometry_list)#,point_size=1.0,line_width=2.0)
    recall = TP_sum/(FN_sum+TP_sum)
    precision = TP_sum/(TP_sum+FP_sum)
    t_error = t_sum/TP_sum
    a_sum = a_sum/TP_sum
    t_median = np.array(t_median)
    a_median = np.array(a_median)
    t_median = np.median(t_median)
    a_median = np.median(a_median)
    print("@20:")
    print(recall)
    print(precision)
    print(t_error)
    print(a_sum)
    print(t_median)
    print(a_median)


def aliagn_box(boxes,tf_matrix):
    #aliagn box in scan with ref
    #output: np.array [N,7]
    if boxes is not None and not isinstance(boxes, np.ndarray):
        boxes = boxes.cpu().numpy()  #(N,7)
    rotation = np.array([tf_matrix[0],tf_matrix[1],tf_matrix[2],
                 tf_matrix[4],tf_matrix[5],tf_matrix[6],
                 tf_matrix[8],tf_matrix[9],tf_matrix[10]]).reshape(3,3)

    trans = np.array([tf_matrix[12],tf_matrix[13],tf_matrix[14]]).reshape(1,3)

    boxes[:,3:6] = boxes[:,3:6] @ rotation# + trans

    return boxes


# def extract_matrix(data_dict):
#
#     return data_dict['scan_dict'][0]['cam_trans']

def assign_target_eval(gt_boxes_inquire,pred_boxes,pred_labels,cam_matrix,trans_ls):
    # A match system for tracking

    # output: target boxes in inquire order
    #         np.array [N,] xyz+size+angle
    #gt_boxes_inquire: np.array(N,3+3+1+1)
    #pred_boxes: np.array(M,3+3+1)
    #pred_labels: np.array(M,)

    N = gt_boxes_inquire.shape[0]

    FN = 0
    TP = 0
    FP = 0
    t_sum = 0
    angle_sum=0
    t_ls = []
    a_ls = []

    for i in range(N):
        sizes = gt_boxes_inquire[i,3:6]

        #Filter One: The same label
        lb = gt_boxes_inquire[i,-1]
        mask = np.where(pred_labels == lb)
        if mask[0].size ==0:
            FN +=1
        else:
            pred_boxes_masked = pred_boxes[mask,:].reshape(-1,7)

            #Filter Two: The nearst distance for size
            dist2 = np.linalg.norm(pred_boxes_masked[:,3:6]-sizes,axis=1)
            target_box = pred_boxes[dist2.argmin(),:]

            pred_trans = compute_trans(target_box,gt_boxes_inquire[i,:],cam_matrix)

            pass_or_not, t, angle_error = satisfy_req(pred_trans,trans_ls[i])

            if pass_or_not:
                TP +=1
                t_sum +=t
                t_ls.append(t)
                a_ls.append(angle_error)
                angle_sum += angle_error
                print("yes!")
            else:
                FP +=1

    return TP,FN,FP,t_sum,angle_sum,t_ls,a_ls


def satisfy_req(pred_trans,trans_ls):
    from scipy.spatial.transform import Rotation as R

    #True: satifies the relocalization requirement
    R_inv = np.linalg.inv(pred_trans)
    #r_pred = R.from_matrix(pred_trans[:3,:3])
    #xyz_pred = r_pred.as_euler('zxy',degrees=True)

    cam_matrix = trans_ls
    R_GT = np.array([cam_matrix[0],cam_matrix[4],cam_matrix[8],cam_matrix[12],
                 cam_matrix[1],cam_matrix[5],cam_matrix[9],cam_matrix[13],
                 cam_matrix[2],cam_matrix[6],cam_matrix[10],cam_matrix[14],
                 0,0,0,1]).reshape(4,4)
    R_delta = R_inv@R_GT

    r = R.from_matrix(R_delta[:3,:3])
    xyz_angle = r.as_euler('zxy',degrees=True) #np.array[1,3]



    if abs(pred_trans[0,3] - trans_ls[12])<0.2 and abs(pred_trans[1,3] - trans_ls[13])<0.2 and abs(pred_trans[2,3] - trans_ls[14])<0.2 :
        if (np.abs(xyz_angle[0])%90) <= 20:
            t_error = np.mean([abs(pred_trans[0,3] - trans_ls[12]),abs(pred_trans[1,3] - trans_ls[13]),abs(pred_trans[2,3] - trans_ls[14])])
            angle_error = np.abs(xyz_angle[0])%20
            return True, t_error, angle_error
        else:
            return False, 1, 1
    else:
        return False,1,1

def satisfy_req2(pred_trans,trans_ls):
    #True: satifies the relocalization requirement
    if (abs(pred_trans[0,3] - trans_ls[12]) + abs(pred_trans[1,3] - trans_ls[13]) + abs(pred_trans[2,3] - trans_ls[14]))<=0.6:
        return True
    else:
        return False

def compute_trans(target_box,gt_boxes_inquire,cam_matrix):
    #target_box:[,7]
    #gt_boxes_inquire:[,8]
    angle = target_box[6] - gt_boxes_inquire[6]
    dx= target_box[3] - gt_boxes_inquire[3]
    dy = target_box[4] - gt_boxes_inquire[4]
    dz = target_box[5] - gt_boxes_inquire[5]
    local_Trans = np.array([np.cos(angle),-np.sin(angle),0,dx,
         np.sin(angle),np.cos(angle),0,dy,
         0,0,1,dz,
         0,0,0,1]).reshape(4,4)

    cam_matrix = np.array([cam_matrix[0],cam_matrix[4],cam_matrix[8],cam_matrix[12],
                 cam_matrix[1],cam_matrix[5],cam_matrix[9],cam_matrix[13],
                 cam_matrix[2],cam_matrix[6],cam_matrix[10],cam_matrix[14],
                 0,0,0,1]).reshape(4,4)

    Trans = local_Trans @ cam_matrix
    return Trans


def extract_info(data_dict):
    #extract inquire ids and cam_trans and ref_if
    #output: ls(len=16), list(len=N), ls, string
    id_list = []
    trans_list = []
    obj_list = data_dict['scan_dict'][0]['obj_movement']
    for i in range(len(obj_list)):
        id = obj_list[i]['instance_reference']
        trans = obj_list[i]['transform']
        id_list.append(id)
        trans_list.append(trans)

    ref_id = data_dict['scan_dict'][0]['ref_id']
    tf_matrix = data_dict['scan_dict'][0]['cam_trans']
    return tf_matrix, id_list, trans_list, ref_id

def extract_gt_boxes(ref_id, inquire_id_ls):
    #output: gt_boxes of inquire id
    #          np.array [N,3+3+1+1]  xyz+size+angle+cls
    data_dir = DATA_PATH + ref_id
    NUM_OBJ = len(inquire_id_ls)
    gt_boxes = []

    with open(data_dir + "/semseg.v2.json", 'r') as load_f:
        load_dict = json.load(load_f)
        seg_groups = load_dict['segGroups']

        idxes_rio7, centroids, sizes, orientation, heading_angles, box_label_mask, objs_id = [(0)] * NUM_OBJ,\
                                                                                             [(-1000, -1000, -1000)] * NUM_OBJ, [(0, 0, 0)] * NUM_OBJ, \
                                                                                             [(0, 0, 0, 0, 0, 0, 0, 0,0)] * NUM_OBJ, [( 0)] * NUM_OBJ, np.zeros(NUM_OBJ), \
                                                                                             [(0)] * NUM_OBJ  # idxes_rio7 are idx in rio7

        #         vote_label_mask = np.zeros((choices.shape[0],2)) #shape:[sample_size,2]
        #         point_votes = np.zeros((choices.shape[0],3)) # shape: sample_size,3. corresponding to "vote-label" in VoteNet

        #count = 0
        for i in range(len(seg_groups)):

            obj_id = load_dict['segGroups'][i]['objectId']
            label = load_dict['segGroups'][i]['label']
            idx_rio7 = label2idx(label)

            if obj_id in inquire_id_ls and idx_rio7!=0:

                obb = load_dict['segGroups'][i]['obb']
                orientation_matrix = obb['normalizedAxes']
                heading_angle = np.arccos(orientation_matrix[0])
                if orientation_matrix[1] < 0:
                    heading_angle = -heading_angle
                if heading_angle >= np.pi:
                    heading_angle = 0

                heading_angle = heading_angle.item() # conver nparray back to scalar



                gt_box = [obb['centroid'][0], obb['centroid'][1], obb['centroid'][2],
                          obb['axesLengths'][0], obb['axesLengths'][2], obb['axesLengths'][1],
                          heading_angle,idx_rio7]

                gt_boxes.append(gt_box)


        return np.array(gt_boxes)

if __name__ == '__main__':
    main()
