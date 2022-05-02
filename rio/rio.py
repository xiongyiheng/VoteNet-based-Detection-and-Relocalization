import os

import open3d as o3d

import numpy as np

import json

from torch.utils.data import Dataset

import pandas as pd

EXCEL_PATH = os.getcwd() + "/rio/config/Classes.xlsx"

MEAN_SIZE_PATH = os.getcwd() + "/rio/config/mean_size.npy"

# load the class mapping excel file
df = pd.read_excel(EXCEL_PATH, sheet_name='Mapping')

MAX_NUM_OBJ = 69

def label2idx(label):
    #input: global label : "sofa"
    #output: index in rio7: 1
    return df[df['Label']==label ]['RIO7 Index'].iloc[0]

def get_pcd_from_ply(path, sample_size):
    # input is the parent path of the .ply file
    # output is all point cloud in format np.asarray with the shape of N * 3
    pcd = o3d.io.read_point_cloud(path + "/labels.instances.annotated.v2.ply")
    pcd_points = np.asarray(pcd.points)
    if pcd_points.shape[0] >= sample_size:
        choices = np.random.choice(pcd_points.shape[0], sample_size, replace=False)
    else:
        choices = np.random.choice(pcd_points.shape[0], sample_size, replace=True)
    return pcd_points[choices], choices

def compute_vote(path,choices,pcd,centroids):
    with open(path + "/semseg.v2.json",'r') as load_f:
        load_dict = json.load(load_f)

        vote_label_mask = np.zeros(choices.shape[0]) # shape:[sample_size,1]
        point_votes = np.zeros((choices.shape[0],3)) # shape: sample_size,3. corresponding to "vote-label" in VoteNet
        seg_groups = load_dict['segGroups']
        for i in range(len(seg_groups)):
            if i >= MAX_NUM_OBJ:
                break
            label = load_dict['segGroups'][i]['label']
            idx_rio7 = label2idx(label)
            obb = load_dict['segGroups'][i]['obb']
            centroid = obb['centroid']
            if idx_rio7 != 0 and 1000 > centroid[0] > -1000 and 1000 > centroid[1] > -1000 and 1000 > centroid[2] > - 1000:
                centroid = centroids[i]
                # the following computing votes need after augmentation
                # get the vote-label-mask
                segments = load_dict['segGroups'][i]['segments']
                segments = np.array(segments)
                samp_mask = np.where(np.in1d(choices, segments))[0] # [sample_size,] :find the index of elements in array A that also appear in array B
                vote_label_mask[samp_mask] = 1.0      # mask[:,0]: this point votes or not

                # get the vote label
                point_votes[samp_mask,:] = np.array(centroid) - pcd[samp_mask,:3]

        #point_votes = np.tile(point_votes,(1,3))

    return vote_label_mask, point_votes

def extract_label(path):
    # input is the parent path of the .json file
    # output: list of semantic labels, centroids, sizes, heading angles
    with open(path + "/semseg.v2.json", 'r') as load_f:
        load_dict = json.load(load_f)

        # with open(path+"/scans.json",'r') as load_scan:
        #     load_scan_dict = json.load(load_scan)

        idxes_rio7, centroids, sizes, orientation, heading_angles, box_label_mask,objs_id = [(0)] * MAX_NUM_OBJ, [
            (-1000, -1000, -1000)] * MAX_NUM_OBJ, [(0, 0, 0)] * MAX_NUM_OBJ, [(0, 0, 0, 0, 0, 0, 0, 0,
                                                                               0)] * MAX_NUM_OBJ, [
                                                                                        (0)] * MAX_NUM_OBJ, np.zeros(
            MAX_NUM_OBJ),[(0)]*MAX_NUM_OBJ # idxes_rio7 are idx in rio7

        #         vote_label_mask = np.zeros((choices.shape[0],2)) #shape:[sample_size,2]
        #         point_votes = np.zeros((choices.shape[0],3)) # shape: sample_size,3. corresponding to "vote-label" in VoteNet

        seg_groups = load_dict['segGroups']
        for i in range(len(seg_groups)):
            label = load_dict['segGroups'][i]['label']
            idx_rio7 = label2idx(label)
            idxes_rio7[i] = idx_rio7
            obb = load_dict['segGroups'][i]['obb']
            centroid = obb['centroid']
            obj_id = load_dict['segGroups'][i]['objectId']
            objs_id[i]=obj_id


            if idx_rio7 != 0 and centroid[0] < 100 and centroid[0] > -100 and centroid[1] < 100 and centroid[
                1] > -100 and centroid[2] < 100 and centroid[2] > -100:
                centroids[i] = centroid
                size = obb['axesLengths']
                sizes[i] = size

                # add heading_angel
                orientation_matrix = obb['normalizedAxes']
                orientation[i] = orientation_matrix
                heading_angle = np.arccos(orientation_matrix[0])
                if orientation_matrix[1] < 0:
                    heading_angle = -heading_angle
                if heading_angle >= np.pi:
                    heading_angle = 0
                heading_angles[i] = heading_angle
                box_label_mask[i] = 1
            else:
                idxes_rio7[i] = 0

    return np.array(centroids), np.array(sizes), np.array(idxes_rio7), np.array(orientation), np.array(
        heading_angles), box_label_mask,np.array(objs_id)#, load_scan_dict

def theta_array_to_class(theta_array):
    theta_class = np.array((theta_array + np.pi)/(np.pi/6),dtype=np.int)
    return theta_class

class RIO(Dataset):
    def __init__(self, is_train, dataset_path, sample_size, is_augment, use_height=True):
        # is_train: training set or val set
        # is_augment: augment or not
        # use_height: use height features as the 4th column of pcd. Ref: https://github.com/facebookresearch/votenet/blob/main/scannet/scannet_detection_dataset.py
        super(RIO, self).__init__()

        self.is_train = is_train
        self.dataset_path = dataset_path
        self.sample_size = sample_size
        self.is_augment = is_augment
        self.use_height = use_height

        if self.is_train:
            f = open(self.dataset_path + "viz.txt", "r")
        else:
            f = open(self.dataset_path + "viz.txt", "r")

        self.index = f.read().splitlines()

        # load the mean size
        MEAN_SIZE = np.load(MEAN_SIZE_PATH)
        self.mean_size = MEAN_SIZE

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        ### return: pcd:x,y,z,height, controid:x,y,z, sizes:x,y,z, idxes: 0-7, mean_size:x,y,z, 
        idx_str = self.index[idx]
        scene_path = self.dataset_path + idx_str

        # extract one scan
        scan, choices = get_pcd_from_ply(scene_path, self.sample_size)

        # extract labels & orientation matrix
        centroids, sizes, idxes_rio7, orientation, heading_angles, box_label_mask,_= extract_label(scene_path)

        # get mean_sizes
        mean_sizes = np.zeros((len(idxes_rio7), 3))
        for i in range(len(idxes_rio7)):
            mean_sizes[i]= self.mean_size[idxes_rio7[i]]

        # get mean_size_array
        mean_size_array = self.mean_size + 0

        # data aug
        if self.is_augment:
            #if np.random.random() > 0.5:
                # Flipping along the YZ plane
                #scan[:,0] = -1 * scan[:,0]
                #centroids[:,0] = -1 * centroids[:,0]
                #heading_angles[heading_angles>0.0] = np.pi - heading_angles[heading_angles>0.0]
                #heading_angles[heading_angles<0.0] = -(np.pi + heading_angles[heading_angles<0.0])

            #if np.random.random() > 0.5:
                # Flipping along the XZ plane
                #scan[:,1] = -1 * scan[:,1]
                #centroids[:,1] = -1 * centroids[:,1]
                #heading_angles = -heading_angles

            # Rotation along up-axis/Z-axis
            heading_angles = np.array(heading_angles, dtype=np.float64)
            #theta = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
            theta = (np.random.random()*np.pi/18) - np.pi/36 # -5 ~ +5 degree
            check_theta = heading_angles + theta
            mask = np.zeros(check_theta.shape)
            mask[check_theta>=np.pi] = 1
            mask[check_theta<-np.pi] = 1
            if np.sum(mask) > 0:
                theta = 0.0
            matrix = np.zeros((3,3))
            c = np.cos(theta)
            s = np.sin(theta)
            matrix[0, 0] = c
            matrix[0, 1] = -s
            matrix[1, 0] = s
            matrix[1, 1] = c
            matrix[2, 2] = 1.0
            scan[:,0:3] = np.dot(scan[:,0:3], np.transpose(matrix))
            centroids[:,0:3] = np.dot(centroids[:,0:3], np.transpose(matrix))
            heading_angles += theta

            # Rescale randomly by 0.9 - 1.1
            proportion = np.random.uniform(0.9, 1.1, 1)
            scan = scan * proportion
            centroids = centroids * proportion
            sizes = sizes * proportion

        # compute vote after data-aug
        vote_label_mask, vote_label = compute_vote(scene_path,choices,scan,centroids)

        # compute the height features     
        if self.use_height:
            floor_height = np.percentile(scan[:,2], 0.99)
            height = scan[:,2] - floor_height
            scan = np.concatenate([scan,np.expand_dims(height, 1)], 1)

        # get heading angle bins and heading residuals
        heading_cls = theta_array_to_class(heading_angles)
        heading_residuals = heading_angles - (np.pi / 12 + np.pi / 6 * (heading_cls - 6))

        # get size residuals
        size_residuals = sizes - mean_sizes # 147, 3 - 147, 3

        ret_dict = {}
        ret_dict['point_clouds'] = scan.astype(np.float32)
        ret_dict['center_label'] = centroids.astype(np.float32)
        ret_dict['heading_class_label'] = heading_cls.astype(np.int64)
        ret_dict['heading_residual_label'] = heading_residuals.astype(np.float32)
        ret_dict['size_class_label'] = idxes_rio7.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = idxes_rio7
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = box_label_mask.astype(np.float32)
        ret_dict['vote_label'] = vote_label.astype(np.float32)
        ret_dict['vote_label_mask'] = vote_label_mask.astype(np.int64)
        #ret_dict['scan_dict'] = scan_dict
        # ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        # ret_dict['pcl_color'] = pcl_color
        return ret_dict
