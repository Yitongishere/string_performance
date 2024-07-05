import argparse
import itertools
import json
import os.path
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from mpl_toolkits.mplot3d.art3d import Line3DCollection

CAM_DICT = {
    'cam0': 21334181,
    'cam1': 21334237,
    'cam2': 21334180,
    'cam3': 21334209,
    'cam4': 21334208,
    'cam5': 21334186,
    'cam6': 21293326,
    'cam7': 21293325,
    'cam8': 21293324,
    'cam9': 21334206,
    'cam10': 21334220,
    'cam11': 21334183,
    'cam12': 21334207,
    'cam13': 21334191,
    'cam14': 21334184,
    'cam15': 21334238,
    'cam16': 21334221,
    'cam17': 21334219,
    'cam18': 21334190,
    'cam19': 21334211,
    'cam20': 21334218,
    'cam21': 21334182,
    'cam22': 21334236,
    'cam23': 21334210
}

"""cello keypoint dict define"""
CELLO_DICT = {
    'scroll_top': 0,
    'nut_l': 1,
    'nut_r': 2,
    'neck_bottom_l': 3,
    'neck_bottom_r': 4,
    'bridge_l': 5,
    'bridge_r': 6,
    'tail_gut': 7,
    'end_pin': 8,
    'frog': 9,
    'tip_plate': 10,
}

"""skeleton define"""
HUMAN_LINKS = [[15, 13],
               [13, 11],
               [16, 14],
               [14, 12],
               [11, 12],
               [5, 11],
               [6, 12],
               [5, 6],
               [5, 7],
               [6, 8],
               [7, 9],
               [8, 10],
               [1, 2],
               [0, 1],
               [0, 2],
               [1, 3],
               [2, 4],
               [3, 5],
               [4, 6],
               [15, 17],
               [15, 18],
               [15, 19],
               [16, 20],
               [16, 21],
               [16, 22],
               [91, 92],
               [92, 93],
               [93, 94],
               [94, 95],
               [91, 96],
               [96, 97],
               [97, 98],
               [98, 99],
               [91, 100],
               [100, 101],
               [101, 102],
               [102, 103],
               [91, 104],
               [104, 105],
               [105, 106],
               [106, 107],
               [91, 108],
               [108, 109],
               [109, 110],
               [110, 111],
               [112, 113],
               [113, 114],
               [114, 115],
               [115, 116],
               [112, 117],
               [117, 118],
               [118, 119],
               [119, 120],
               [112, 121],
               [121, 122],
               [122, 123],
               [123, 124],
               [112, 125],
               [125, 126],
               [126, 127],
               [127, 128],
               [112, 129],
               [129, 130],
               [130, 131],
               [131, 132]]

HUMAN_WITHOUT_HANDS_LINKS = [[15, 13],
                             [13, 11],
                             [16, 14],
                             [14, 12],
                             [11, 12],
                             [5, 11],
                             [6, 12],
                             [5, 6],
                             [5, 7],
                             [6, 8],
                             [7, 9],
                             [8, 10],
                             [1, 2],
                             [0, 1],
                             [0, 2],
                             [1, 3],
                             [2, 4],
                             [3, 5],
                             [4, 6],
                             [15, 17],
                             [15, 18],
                             [15, 19],
                             [16, 20],
                             [16, 21],
                             [16, 22]]

HANDS_LINKS = [[91, 92],
               [92, 93],
               [93, 94],
               [94, 95],
               [91, 96],
               [96, 97],
               [97, 98],
               [98, 99],
               [91, 100],
               [100, 101],
               [101, 102],
               [102, 103],
               [91, 104],
               [104, 105],
               [105, 106],
               [106, 107],
               [91, 108],
               [108, 109],
               [109, 110],
               [110, 111],
               [112, 113],
               [113, 114],
               [114, 115],
               [115, 116],
               [112, 117],
               [117, 118],
               [118, 119],
               [119, 120],
               [112, 121],
               [121, 122],
               [122, 123],
               [123, 124],
               [112, 125],
               [125, 126],
               [126, 127],
               [127, 128],
               [112, 129],
               [129, 130],
               [130, 131],
               [131, 132]]

HANDS_WITHOUT_LH_THUMB_LINKS = [[91, 96],
                                [96, 97],
                                [97, 98],
                                [98, 99],
                                [91, 100],
                                [100, 101],
                                [101, 102],
                                [102, 103],
                                [91, 104],
                                [104, 105],
                                [105, 106],
                                [106, 107],
                                [91, 108],
                                [108, 109],
                                [109, 110],
                                [110, 111],
                                [112, 113],
                                [113, 114],
                                [114, 115],
                                [115, 116],
                                [112, 117],
                                [117, 118],
                                [118, 119],
                                [119, 120],
                                [112, 121],
                                [121, 122],
                                [122, 123],
                                [123, 124],
                                [112, 125],
                                [125, 126],
                                [126, 127],
                                [127, 128],
                                [112, 129],
                                [129, 130],
                                [130, 131],
                                [131, 132]]

HUMAN_WITHOUT_LH_THUMB_LINKS = [[15, 13],
                                [13, 11],
                                [16, 14],
                                [14, 12],
                                [11, 12],
                                [5, 11],
                                [6, 12],
                                [5, 6],
                                [5, 7],
                                [6, 8],
                                [7, 9],
                                [8, 10],
                                [1, 2],
                                [0, 1],
                                [0, 2],
                                [1, 3],
                                [2, 4],
                                [3, 5],
                                [4, 6],
                                [15, 17],
                                [15, 18],
                                [15, 19],
                                [16, 20],
                                [16, 21],
                                [16, 22],
                                [91, 96],
                                [96, 97],
                                [97, 98],
                                [98, 99],
                                [91, 100],
                                [100, 101],
                                [101, 102],
                                [102, 103],
                                [91, 104],
                                [104, 105],
                                [105, 106],
                                [106, 107],
                                [91, 108],
                                [108, 109],
                                [109, 110],
                                [110, 111],
                                [112, 113],
                                [113, 114],
                                [114, 115],
                                [115, 116],
                                [112, 117],
                                [117, 118],
                                [118, 119],
                                [119, 120],
                                [112, 121],
                                [121, 122],
                                [122, 123],
                                [123, 124],
                                [112, 125],
                                [125, 126],
                                [126, 127],
                                [127, 128],
                                [112, 129],
                                [129, 130],
                                [130, 131],
                                [131, 132]]

LH_THUMB_LINKS = [[91, 92],
                  [92, 93],
                  [93, 94],
                  [94, 95]]

LEFT_HAND_LINKS = [[5, 7],
                   [7, 9],
                   [91, 92],
                   [92, 93],
                   [93, 94],
                   [94, 95],
                   [91, 96],
                   [96, 97],
                   [97, 98],
                   [98, 99],
                   [91, 100],
                   [100, 101],
                   [101, 102],
                   [102, 103],
                   [91, 104],
                   [104, 105],
                   [105, 106],
                   [106, 107],
                   [91, 108],
                   [108, 109],
                   [109, 110],
                   [110, 111]]

CELLO_LINKS = [[133, 134],
               [133, 135],
               [134, 136],
               [135, 137],
               [136, 138],
               [137, 138],
               [138, 139]]

CELLO_FINGERBOARD_LINKS = [[142, 144],
                           [144, 146],
                           [146, 148],
                           [143, 145],
                           [145, 147],
                           [147, 149]]

BOW_LINKS = [[140, 141]]

STRING_LINKS = [[142, 143],
                [144, 145],
                [146, 147],
                [148, 149]]

LEFT_HAND_THUMB_LINKS = [
    [91, 92],
    [92, 93],
    [93, 94],
    [94, 95]
]

LEFT_HAND_WITHOUT_THUMB_LINKS = [
    # [5, 7],
    [91, 96],
    [96, 97],
    [97, 98],
    [98, 99],
    [91, 100],
    [100, 101],
    [101, 102],
    [102, 103],
    [91, 104],
    [104, 105],
    [105, 106],
    [106, 107],
    [91, 108],
    [108, 109],
    [109, 110],
    [110, 111]]

LEFT_ARM_LINKS = [[7, 9]]

USED_FINGER_LINKS = [[151, 152],
                     [152, 153],
                     [153, 154]]

FULL_FINGER_INDICES = [[96, 97, 98, 99],
                       [100, 101, 102, 103],
                       [104, 105, 106, 107],
                       [108, 109, 110, 111]]


def get_all_combinations(cams):
    cams_num = len(cams)
    cam_coms = []
    for i in range(2, cams_num):
        i_com = list(itertools.combinations(cams, i))
        cam_coms += i_com
    return cam_coms


def make_projection_matrix(calibration_data, cams=["cam0", "cam1", "cam2"],
                           rm=None, offset=None):
    """use the calibration data to compute the projection matrix for cameras
    Returns:
        projection_matrices: array [num_cam, 3, 4] projection matrix
        rm: the rotation matrix to make x-y the floor, [3, 3]
        offset: the offset of camera, [3, 1]
    """
    projection_matrices = []
    for cam in cams:
        cam_matrix = calibration_data[cam]['K']
        cam_matrix = np.array(cam_matrix).reshape([3, 3])
        world_location = np.array(calibration_data[cam]['T']).reshape(3, 1)
        world_orientation = np.array(calibration_data[cam]['R']).reshape(3, 3)
        R0 = world_orientation.copy()
        if rm is not None:
            rm_inv = np.linalg.inv(rm)
            world_orientation = world_orientation @ rm_inv
            # world_location = rm_inv @ world_location
        if offset is not None:
            # ic(cam, world_location)
            # ic(offset)
            # ic(world_orientation@offset)
            world_location = R0 @ offset.reshape([3, 1]) + world_location
            # ic(cam, offset, world_location)
        # print(cam_matrix, np.concatenate([world_orientation, world_location], axis=1))
        projection_matrix = np.matmul(cam_matrix, np.concatenate([world_orientation, world_location], axis=1))
        projection_matrices.append(projection_matrix)
    projection_matrices = np.array(projection_matrices)
    return projection_matrices


def triangulate(image_coordinates, projection_matrices):
    '''
    The base triangulation function for NCams. Takes image coordinates and projection matrices from
    2+ cameras and will produce a triangulated point with the desired approach.

    Arguments:
        image_coordinates {array or list of} -- the x,y coordinates of a given marker for multiple
            cameras. The points must be in the format (1,2) if in a list or (n,2) if an array.
        projection_matrices {list} -- the projection matrices for the cameras corresponding
        to each image points input.

    Keyword Arguments:
        mode {str} -- the triangulation method to use:
            full_rank - performs SVD to find the point with the least squares error between all
                projection lines. If a threshold is given along with confidence values then only
                points above the threshold will be used.
            best_n - uses the n number of cameras with the highest confidence values for the
                triangulation. If a threshold is given then only points above the threshold will
                be considered.
            cluster - [in development] performs all combinations of triangulations and checks for
                outlying points suggesting erroneous image coordinates from one or more cameras.
                After removing the camera(s) that produce out of cluser points it then performs the
                full_rank triangulation.
        confidence_values {list or array} -- the confidence values for the points given by the
            marking system (e.g. DeepLabCut)
        threshold {float} -- the minimum confidence value to accept for triangulation.

    Output:
        u_3d {(1,3) np.array} -- the triangulated point produced.

    '''
    u_3d = np.zeros((1, 3))
    u_3d.fill(np.nan)

    # ic(image_coordinates)

    # Check if image coordinates are formatted properly
    if isinstance(image_coordinates, list):
        if len(image_coordinates) > 1:
            image_coordinates = np.vstack(image_coordinates)
            # ic(image_coordinates)
        else:
            ic('Return nan directly')
            return u_3d

    if not np.shape(image_coordinates)[1] == 2:
        raise ValueError('ncams.reconstruction.triangulate only accepts numpy.ndarrays or lists of' +
                         'in the format (camera, [x,y])')

    num_cameras = np.shape(image_coordinates)[0]
    if num_cameras < 2:  # return NaNs if insufficient points to triangulate
        return u_3d

    if num_cameras != len(projection_matrices):
        raise ValueError('Different number of coordinate pairs and projection matrices given.')

    decomp_matrix = np.empty((num_cameras * 2, 4))
    for decomp_idx in range(num_cameras):
        point_mat = image_coordinates[decomp_idx]
        projection_mat = projection_matrices[decomp_idx]

        temp_decomp = np.vstack([
            [point_mat[0] * projection_mat[2, :] - projection_mat[0, :]],
            [point_mat[1] * projection_mat[2, :] - projection_mat[1, :]]])

        decomp_matrix[decomp_idx * 2:decomp_idx * 2 + 2, :] = temp_decomp

    Q = decomp_matrix.T.dot(decomp_matrix)
    u, _, _ = np.linalg.svd(Q)
    u = u[:, -1, np.newaxis]
    u_3d = np.transpose((u / u[-1, :])[0:-1, :])

    return u_3d


def triangulate_joints(keypoints_mview, projection_matrices, num_joint, kpt_thr=0.5):
    """
    perform triangulation on the multiview mmpose estimation results for a frame
    keypoints_mview: [num_cams, num_joints, 3], [x, y, score]
    projection_matrices: [num_cams, 3, 4]
    returns: keypoints_3d [num_joints, 3]
    """
    # num_obj = pose_mview[0][0]['keypoints'].shape[0]
    # num_joint = dataset_info.keypoint_num
    keypoints_3d = np.empty([num_joint, 3])
    keypoints_3d.fill(np.nan)
    # keypoints_mview = np.array([pose_mview[i][0]['keypoints'] for i in range(num_cams)])    #[num_cams, num_joints, 3], [x, y, score]
    for j in range(num_joint):
        cams_detected = keypoints_mview[:, j, 2] > kpt_thr
        # ic(cams_detected)
        cam_idx = np.where(cams_detected)[0]
        # ic(np.where(cams_detected)[0])
        if np.sum(cams_detected) < 2:
            continue
        # ic(cam_idx)
        u_3d = triangulate(keypoints_mview[cam_idx, j, :2], projection_matrices[cam_idx])
        keypoints_3d[j, :] = u_3d
    return keypoints_3d


def ransac_triangulate_joints(keypoints_mview, projection_matrices, num_kpt, epsilon=150, kpt_thr=0.6):
    """
    perform ransac triangulation on the multiview mmpose estimation results for a frame
    keypoints_mview: [num_cams, num_kpt, 3], [x, y, score]
    projection_matrices: [num_cams, 3, 4]
    returns: keypoints_3d [num_joints, 3]
    """

    keypoints_3d = np.empty([num_kpt, 3])
    keypoints_3d.fill(np.nan)
    num_cams = keypoints_mview.shape[0]

    for j in range(num_kpt):

        cams_detected = keypoints_mview[:, j, 2] > kpt_thr
        # ic(cams_detected)
        cam_idx = np.where(cams_detected)[0]
        # ic(cam_idx)
        if np.sum(cams_detected) < 2:
            continue
        cam_combinations = list(itertools.combinations(cam_idx, 2))
        inlier_set = set()
        for i in cam_combinations:
            # use the minimum cam to estimate the model
            # sampled_cam = sorted(random.sample(cam_set, 2))
            sampled_cam = sorted(list(i))

            kp3d = triangulate(keypoints_mview[sampled_cam, j, :2], projection_matrices[sampled_cam])
            # convert from Euclidean Coordinates to Homogeneous Coordinates
            ones = np.ones((1, 1))
            kp4d = np.concatenate([kp3d, ones], axis=1)  # shape: (1, 4)
            kp4d = kp4d.reshape(-1)  # shape: (4, 1)
            # reprojection: p = mP
            kp2d = np.matmul(projection_matrices, kp4d)  # shape: (num_cams, 3, 4) dot (4, 1) = (num_cams, 3, 1)
            kp2d = kp2d.reshape((num_cams, 3))  # shape: (num_cams, 3)
            kp2d = kp2d / kp2d[:, 2:3]  # shape: (num_cams, 3)
            points_2d_eu = keypoints_mview[:, j, :2]  # shape: (num_cams, 2)
            # convert from Euclidean Coordinates to Homogeneous Coordinates
            ones = np.ones((num_cams, 1))
            points_2d_ho = np.concatenate([points_2d_eu, ones], axis=1)  # shape: (num_cams, 3)
            reprojection_error = np.sqrt(np.sum((kp2d - points_2d_ho) ** 2, axis=1))
            # compute the new inlier list whose re-projection error is smaller than the threshold
            new_inlier_set = set([i for i, v in enumerate(reprojection_error) if v < epsilon])
            # leave the new inlier set IFF it has more cams
            if len(new_inlier_set) > len(inlier_set):
                inlier_set = new_inlier_set
            if len(inlier_set) == num_cams:
                break
        if len(inlier_set) < 2:
            inlier_set = sampled_cam.copy()

        inlier_list = sorted(list(inlier_set))
        # ic(j, inlier_list)
        kp3d = triangulate(keypoints_mview[inlier_list, j, :2], projection_matrices[inlier_list])
        keypoints_3d[j, :] = kp3d

    return keypoints_3d


def compute_axis_lim(triangulated_points, scale_factor=1):
    # ic(triangulated_points.shape)
    # triangulated_points in shape [num_frame, num_keypoint, 3 axis]
    xlim, ylim, zlim = None, None, None
    # ic(triangulated_points.shape)
    minmax = np.nanpercentile(triangulated_points, q=[0, 100], axis=0).T
    # ic(minmax)
    minmax *= 1.
    minmax_range = (minmax[:, 1] - minmax[:, 0]).max() / 2
    if xlim is None:
        mid_x = np.mean(minmax[0])
        xlim = mid_x - minmax_range / scale_factor, mid_x + minmax_range / scale_factor
    if ylim is None:
        mid_y = np.mean(minmax[1])
        ylim = mid_y - minmax_range / scale_factor, mid_y + minmax_range / scale_factor
    if zlim is None:
        mid_z = np.mean(minmax[2])
        zlim = mid_z - minmax_range / scale_factor, mid_z + minmax_range / scale_factor
    return xlim, ylim, zlim


def visualize_3d(data, proj_path, file_path='tri_3d', view_angle='whole'):
    xlim, ylim, zlim = None, None, None
    framenum = data.shape[0]
    key_points_num = data.shape[1]

    # if not os.path.exists(f'../kp_3d_result/'):
    #     os.makedirs(f'../kp_3d_result/')
    #
    # if not os.path.exists(f'../kp_3d_result/{proj_path}/'):
    #     os.makedirs(f'../kp_3d_result/{proj_path}/')

    if not os.path.exists(f'./kp_3d_result/{proj_path}/{file_path}'):
        os.makedirs(f'./kp_3d_result/{proj_path}/{file_path}', exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'./kp_3d_result/{proj_path}/{file_path}/output_{view_angle}.avi', fourcc, fps=30,
                          frameSize=[1000, 1000])

    zoom_in = view_angle == 'finger'
    for f in range(framenum):
        kp_3d = data[f]
        fig = plt.figure(figsize=[10, 10])
        axes3 = fig.add_subplot(projection="3d", computed_zorder=False)
        if xlim is None:
            if zoom_in:
                string_data = data[f][142:155]
                left_arm_data = data[f][5:10:2]
                string_with_arm = np.concatenate((string_data, left_arm_data), axis=0)
                xlim, ylim, zlim = compute_axis_lim(string_with_arm, scale_factor=1.2)
            else:
                xlim, ylim, zlim = compute_axis_lim(data[f])

        if zoom_in:
            axes3.view_init(azim=90, elev=-40, roll=10)
        else:
            # axes3.view_init(azim=90, elev=-60, roll=0)
            axes3.view_init(azim=133, elev=-16, roll=-45)
        # ic(xlim, ylim, zlim)
        # axes3.set_box_aspect(aspect=[5, 5, 5])

        axes3.set_xlim3d(xlim)
        axes3.set_ylim3d(ylim)
        axes3.set_zlim3d(zlim)
        axes3.set_box_aspect((1, 1, 1))

        human_segs3d = kp_3d[tuple([HUMAN_LINKS])]
        cello_segs3d = kp_3d[tuple([CELLO_LINKS])]
        bow_segs3d = kp_3d[tuple([BOW_LINKS])]
        left_hand_segs3d = kp_3d[tuple([LEFT_HAND_LINKS])]

        if zoom_in:
            if key_points_num > 142:
                string_segs3d = kp_3d[tuple([STRING_LINKS])]
                string_coll_3d = Line3DCollection(string_segs3d, edgecolors='black', linewidths=1, zorder=98)
                axes3.add_collection(string_coll_3d)
            left_hand_coll_3d = Line3DCollection(left_hand_segs3d, linewidths=1, zorder=1)
            axes3.add_collection(left_hand_coll_3d)

            # left hand
            axes3.scatter(kp_3d[91:112, 0],
                          kp_3d[91:112, 1],
                          kp_3d[91:112, 2], s=5, c='#1f77b4', zorder=1)
            # left arm
            axes3.scatter(kp_3d[5:10:2, 0],
                          kp_3d[5:10:2, 1],
                          kp_3d[5:10:2, 2], s=5, c='#1f77b4', zorder=1)
            if key_points_num > 142:
                axes3.scatter(kp_3d[142:150, 0],
                              kp_3d[142:150, 1],
                              kp_3d[142:150, 2], c='black', s=5, zorder=98)
            if True not in np.isnan(kp_3d[150]):
                axes3.scatter(kp_3d[150, 0],
                              kp_3d[150, 1],
                              kp_3d[150, 2], c='r', s=30,
                              zorder=100)  # zorder must be the biggest so that it would not be occluded
                # axes3.scatter(kp_3d[151:155, 0],
                #               kp_3d[151:155, 1],
                #               kp_3d[151:155, 2], c='orange', s=30, zorder=99)

        else:
            human_coll_3d = Line3DCollection(human_segs3d, linewidths=1, zorder=1)
            cello_coll_3d = Line3DCollection(cello_segs3d, edgecolors='saddlebrown', linewidths=1, zorder=1)
            bow_coll_3d = Line3DCollection(bow_segs3d, edgecolors='goldenrod', linewidths=1, zorder=1)

            axes3.add_collection(human_coll_3d)
            axes3.add_collection(cello_coll_3d)
            axes3.add_collection(bow_coll_3d)

            axes3.scatter(kp_3d[0:133, 0],
                          kp_3d[0:133, 1],
                          kp_3d[0:133, 2], s=5, zorder=1)
            axes3.scatter(kp_3d[133:140, 0],
                          kp_3d[133:140, 1],
                          kp_3d[133:140, 2], c='saddlebrown', s=5, zorder=1)
            axes3.scatter(kp_3d[140:142, 0],
                          kp_3d[140:142, 1],
                          kp_3d[140:142, 2], c='goldenrod', s=5, zorder=1)
            if key_points_num > 142:
                axes3.scatter(kp_3d[142:150, 0],
                              kp_3d[142:150, 1],
                              kp_3d[142:150, 2], c='black', s=5, zorder=98)
                string_segs3d = kp_3d[tuple([STRING_LINKS])]
                string_coll_3d = Line3DCollection(string_segs3d, edgecolors='black', linewidths=1, zorder=98)
                axes3.add_collection(string_coll_3d)
                if True not in np.isnan(kp_3d[150]):
                    axes3.scatter(kp_3d[150, 0],
                                  kp_3d[150, 1],
                                  kp_3d[150, 2], c='r', s=5,
                                  zorder=100)  # zorder must be the biggest so that it would not be occluded
                    axes3.scatter(kp_3d[151:155, 0],
                                  kp_3d[151:155, 1],
                                  kp_3d[151:155, 2], c='orange', s=5, zorder=99)

        # axes3.grid(None)
        # axes3.axis('off')
        plt.show()

        plt.savefig(f'./kp_3d_result/{proj_path}/{file_path}/{f}.jpg')

        canvas = fig.canvas
        canvas.draw()
        width, height = canvas.get_width_height()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(height, width, 3)
        image_array = image_array[:, :, ::-1]  # rgb to bgr
        out.write(image_array)

        print(f'{file_path} frame {f} graph generated.')

        plt.close()

# TODO: EDIT KEY POINT NUMBER
KPT_NUM = 142

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='triangulation_pipeline')
    parser.add_argument('--summary_jsonfile',
                        default='../data/cello/cello01/cello01_summary.json',
                        type=str, 
                        required=True)
    parser.add_argument('--instrument', default='cello', type=str, required=True)
    parser.add_argument('--parent_dir', default=None, type=str, required=False)
    parser.add_argument('--proj_dir', default=None, type=str, required=False)
    parser.add_argument('--start_frame', default=None, type=int, required=False)
    parser.add_argument('--end_frame', default=None, type=int, required=False)
    
    args = parser.parse_args()
    instrument = args.instrument
    
    with open(args.summary_jsonfile,'r') as f:
        summary = json.load(f)
    f.close()
    
    if args.parent_dir is None or args.proj_dir is None:
        parent_dir = args.summary_jsonfile.split(os.sep)[-3]
        proj_dir = args.summary_jsonfile.split(os.sep)[-2]
    else:
        parent_dir = args.parent_dir
        proj_dir = args.proj_dir
    
    if args.start_frame is None:
        start_frame = summary['StartFrame']
    else:
        start_frame = args.start_frame
    
    if args.end_frame is None:
        end_frame = summary['EndFrame']
    else:
        end_frame = args.end_frame
    
    cam_param = summary['CameraParameter']

    point_offset = 0
    if instrument == 'violin':
        point_offset += 2  # violin has less key points
        KPT_NUM -= 2
    

    """read 2d results."""
    # cello used cams
    customed_cams = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6',
                 'cam7', 'cam8', 'cam9', 'cam10', 'cam11', 'cam12', 'cam13',
                 'cam14', 'cam15', 'cam16', 'cam17', 'cam18', 'cam19', 'cam20', 'cam21', 'cam22', 'cam23']
    
    if instrument == 'violin':
        customed_cams = ['cam8', 'cam9', 'cam10', 'cam11', 'cam12', 'cam13', 'cam20', 'cam21', 'cam22']
        CELLO_LINKS = CELLO_LINKS[:-3]
        CELLO_LINKS.append([136, 137])
        BOW_LINKS = np.array(BOW_LINKS) - 2
    
    listed_cames = list(cam_param.keys())
    used_cams = sorted(set(listed_cames).intersection(set(customed_cams)),key=lambda s: int(s.split('cam')[1]))
    print(used_cams)
    # used_cams = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6',
    #              'cam7', 'cam11', 'cam12', 'cam13', 'cam14', 'cam15', 'cam16',
    #              'cam17', 'cam18', 'cam19','cam20','cam21','cam22','cam23']

    # start_frame = 128
    # end_frame = 129
    # end_frame = 786
    # end_frame = 190

    kp_3d_all = []
    for ff in range(start_frame, end_frame + 1):
        kp_2d_all_cams = []
        cam_ff = used_cams.copy()
        for cc in used_cams:
            actual_ff = ff
            try:
                frame_drop_path = f'../data/{parent_dir}/{proj_dir}/videos/{CAM_DICT[cc]}_FrameDropIDLog.txt'
                drop_frames = np.array(open(frame_drop_path).readlines(), dtype=int)
                drop_frames = sorted(drop_frames)
                # remove camera that drop frames
                # if ff in drop_frames:
                #     print(f'Remove cam {CAM_DICT[cc]} for frame {ff}!')
                #     cam_ff.remove(cc)
                #     continue
                dropped = 0
                removed = False  # whether the camera is removed
                for drop_frame in drop_frames:
                    if drop_frame < ff:
                        dropped += 1
                    elif drop_frame == ff:
                        print(f'Remove cam {CAM_DICT[cc]} for frame {ff}!')
                        cam_ff.remove(cc)
                        removed = True
                        break
                    elif drop_frame > ff:
                        break
                if removed:
                    continue
                else:
                    actual_ff -= dropped

            except FileNotFoundError as e:
                pass
            try:
                #print(1)
                human_joint = f'../human_kp_2d/kp_result/{parent_dir}/{proj_dir}/{CAM_DICT[cc]}/{actual_ff}.json'
                human_2d_cc_ff = np.array(json.load(open(human_joint)))
            except FileNotFoundError as e:
                # remove camera that has no human joint data
                cam_ff.remove(cc)
                continue
            # TODO: EDIT CELLO KEY POINTS
            cello_2d_cc_ff = np.zeros([9-point_offset, 3])  # 9 cello key points in total (default score 0 will not be used)
            try:
                #print(2)
                cello_json_path = f'../cello_kp_2d/kp_result/{parent_dir}/{proj_dir}/{CAM_DICT[cc]}/{actual_ff}.json'
                cello_keypoints = json.load(open(cello_json_path))
                # Resolve XML
                # for each_ann in labelme['shapes']:  # manually labelled
                #     if each_ann['shape_type'] == 'point':
                #         kpt_label = each_ann['label']
                #         kpt_idx = CELLO_DICT[kpt_label]
                #         kpt_xy = each_ann['points'][0]
                #         cello_2d_cc_ff[kpt_idx] = [kpt_xy[0], kpt_xy[1], 1]  # set the human-label score to be 1
                for idx, cello_keypoint in enumerate(cello_keypoints):
                    cello_2d_cc_ff[idx] = [cello_keypoint[0], cello_keypoint[1], cello_keypoint[2]]
                kp_2d_cc_ff = np.concatenate([human_2d_cc_ff, cello_2d_cc_ff], axis=0)
                kp_2d_all_cams.append(kp_2d_cc_ff)
            except FileNotFoundError as e:
                kp_2d_cc_ff = np.concatenate([human_2d_cc_ff, cello_2d_cc_ff], axis=0)
                kp_2d_all_cams.append(kp_2d_cc_ff)
                continue
        # make projection matrix using filtered camera
        proj_mat = make_projection_matrix(cam_param, cams=cam_ff)
        kp_2d_all_cams = np.array(kp_2d_all_cams)
        # kp_3d = triangulate_joints(kp_2d_all_cams, proj_mat, num_joint=133, kpt_thr=0.6)
        kp_3d = ransac_triangulate_joints(kp_2d_all_cams, proj_mat, num_kpt=KPT_NUM, epsilon=60, kpt_thr=0.6)
        # kp_3d_all[ff] = kp_3d
        kp_3d_all.append(kp_3d)
        print(f'{proj_dir}: | Frame {ff} triangulation done.')

    kp_3d_all = np.array(kp_3d_all)

    # kp_3d_kalman = Kalman_filter(kp_3d_all, KPT_NUM)
    # kp_3d_smooth = Savgol_Filter(kp_3d_all, KPT_NUM)

    # visualize_3d(kp_3d_all, f'{parent_dir}/{proj_dir}')

    if not os.path.exists(f'./kp_3d_result/{parent_dir}/{proj_dir}/'):
        os.makedirs(f'./kp_3d_result/{parent_dir}/{proj_dir}/', exist_ok=True)
    data_dict = {'kp_3d_all_dw': kp_3d_all.tolist()}
    with open(f'./kp_3d_result/{parent_dir}/{proj_dir}/kp_3d_all_dw.json', 'w') as f:
        json.dump(data_dict, f)

    # ffmpeg -r 30 -i sample%d.jpg output.mp4 -crf 0
