import numpy as np
import open3d as o3d



# def get_box_from_proposal(boxes):
#     centroids = proposal[10:13].cpu().detach().numpy()
#     size_cls = torch.max(F.softmax(proposal[13:21], dim=0), dim=0)[1]
#     size_residuals = proposal[21:45]
#     size_residuals = size_residuals[size_cls*3:size_cls*3+3]
#     heading_cls = torch.max(F.softmax(proposal[45:57], dim=0), dim=0)[1]
#     heading_residuals = proposal[57:69]
#     size = MEAN_SIZE[size_cls] + size_residuals.cpu().detach().numpy() * MEAN_SIZE[size_cls]
#     heading_angle = np.pi/6*(heading_cls.cpu()-6)+np.pi/12 + heading_residuals[heading_cls].cpu().detach().numpy()*(np.pi/12)
#     return centroids, size, heading_angle
num_cls = 7

def box_center_to_corner(centroid, size, normalizedAxes):
    # To return
    corner_boxes = np.zeros((8, 3))

    #centroid[[0,1,2]] = centroid[[0,2,1]]
    translation = centroid
    h, w, l = size[1], size[2], size[0]

    # Create a bounding box outline
    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]])

    # Standard 3x3 rotation matrix around the Z axis
    rotation_matrix = np.array([
        [normalizedAxes[0], normalizedAxes[1], 0.0],
        [normalizedAxes[6], normalizedAxes[7], 0.0],
        [0.0, 0.0, 1.0]])

    # Repeat the [x, y, z] eight times
    eight_points = np.tile(translation, (8, 1))

    # Translate the rotated bounding box by the
    # original center position to obtain the final box
    corner_box = np.dot(
        rotation_matrix, bounding_box) + eight_points.transpose()

    return corner_box.transpose()

def get_line_set(centroid, sizes, normalizedAxes, is_predict,ref_labels):
    # visualize bbox in open3d
    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
            [4, 5], [5, 6], [6, 7], [4, 7],
            [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    if True:
        print(ref_labels)
        if ref_labels.item() == 1:
            colors = [[100/255, 149/255, 237/255] for _ in range(len(lines))] #DarkSlateBlue
        if ref_labels.item()==2 :
            colors = [[47 / 255, 79 / 255, 47 / 255] for _ in range(len(lines))]  # DarkGreen
        if ref_labels.item() == 3:
            colors = [[0 / 255, 0 / 255, 0 / 255] for _ in range(len(lines))]  # GreenYellow
        if ref_labels.item() == 4:
            colors = [[255 / 255, 127 / 255, 0 / 255] for _ in range(len(lines))]  # coral
        if ref_labels.item() == 5:
            colors = [[0 / 255, 255 / 255, 0 / 255] for _ in range(len(lines))]  # DarkGreen
        if ref_labels.item() == 6:
            colors = [[255 / 255, 0 / 255, 0 / 255] for _ in range(len(lines))]  # DarkGreen
        if ref_labels.item() == 7:
            colors = [[0 / 255, 0 / 255, 255 / 255] for _ in range(len(lines))]  # DarkGreen
        # elif ref_labels.item()/6 <= 2 :
        #     colors = [[0,ref_labels.item()%6/6, ref_labels.item()%6/6] for _ in range(len(lines))]
        # elif ref_labels.item()/3 <= 3 :
        #     colors = [[ref_labels.item()%8/8,0,ref_labels.item()%8/8] for _ in range(len(lines))]
        #colors = [[ref_labels.item()/num_cls, 0, 0] for _ in range(len(lines))]
    else:
        colors = [[1, 0, 0] for _ in range(len(lines))]

    line_set = o3d.geometry.LineSet()
    corner_box = box_center_to_corner(centroid, sizes, normalizedAxes)
    line_set.points = o3d.utility.Vector3dVector(corner_box)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    #line_set.width =
    return line_set


def get_boxes(boxes,is_predict,geometry_list,ref_labels):
    #input: boxes para: center, size, heading_angle in size[N,7]
    #output: gt_boxes: line_set

    offset_1 = np.array([0.01,0.01,0.01])
    #offset_2 = np.array([-0.01, -0.01, -0.01])
    offset_2 = np.array([0.02, 0.02, 0.02])

    #geometry_list = []
    for i in range(boxes.shape[1]):

        heading_angel = boxes[:,i, -1]#.cpu().detach().numpy()
        c = np.cos(heading_angel)
        s = np.sin(heading_angel)
        orientation = np.array([c, -s, 0.0, 0.0, 0.0, 0.0, s, c, 1.0])
        # if boxes[:,i,2]>2 or (boxes[:,i,0]<-3.5) or (boxes[:,i,0]>3.5) or (boxes[i,1]<-6.5) or (boxes[i,1]>5.5):
        #     continue
        if is_predict:
            line_set = get_line_set(boxes[0,i,0:3], boxes[0,i, 3:6],orientation,is_predict,ref_labels[:,i])
            geometry_list.append(line_set)
            line_set = get_line_set(boxes[0,i,0:3], boxes[0,i, 3:6]+offset_1,orientation,is_predict,ref_labels[:,i])
            geometry_list.append(line_set)
            line_set = get_line_set(boxes[0,i,0:3], boxes[0,i, 3:6]+offset_2,orientation,is_predict,ref_labels[:,i])
            geometry_list.append(line_set)
            line_set = get_line_set(boxes[0,i,0:3], boxes[0,i, 3:6]-offset_1,orientation,is_predict,ref_labels[:,i])
            geometry_list.append(line_set)
            line_set = get_line_set(boxes[0,i,0:3], boxes[0,i, 3:6]-offset_2,orientation,is_predict,ref_labels[:,i])
            geometry_list.append(line_set)
        else:
            line_set = get_line_set(boxes[0, i, 0:3], boxes[0, i, 3:6], orientation, is_predict, ref_labels[:, i])
            geometry_list.append(line_set)
            line_set = get_line_set(boxes[0, i, 0:3], boxes[0, i, 3:6] + offset_1, orientation, is_predict,
                                    ref_labels[:, i])
            geometry_list.append(line_set)
            line_set = get_line_set(boxes[0, i, 0:3], boxes[0, i, 3:6] + offset_2, orientation, is_predict,
                                    ref_labels[:, i])
            geometry_list.append(line_set)
            line_set = get_line_set(boxes[0, i, 0:3], boxes[0, i, 3:6] - offset_1, orientation, is_predict,
                                    ref_labels[:, i])
            geometry_list.append(line_set)
            line_set = get_line_set(boxes[0, i, 0:3], boxes[0, i, 3:6] - offset_2, orientation, is_predict,
                                    ref_labels[:, i])
            geometry_list.append(line_set)
        #geometry_list.append(line_set)

    return geometry_list


def get_pcd_from_np(points,geometry_list):
    pcd = o3d.geometry.PointCloud()
    points = points[:,:,0:3].cpu().detach().numpy().reshape(-1,3)
    pcd.points = o3d.utility.Vector3dVector(points[:,0:3])
    colors = np.zeros_like(points)
    colors[:,:3] = [0.7,0.7,0.7]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    geometry_list.append(pcd)

    return geometry_list

