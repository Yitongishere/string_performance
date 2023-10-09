import os.path
import random
import xml.dom.minidom

import cv2
from icecream import ic
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import itertools


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


def triangulate_joints(keypoints_mview, projection_matrices, num_joint, kpt_thr):
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


def ransac_triangulate_joints(keypoints_mview, projection_matrices, num_joint, niter=50, epsilon=150, kpt_thr=0.6):
    """
    perform ransac triangulation on the multiview mmpose estimation results for a frame
    keypoints_mview: [num_cams, num_joints, 3], [x, y, score]
    projection_matrices: [num_cams, 3, 4]
    returns: keypoints_3d [num_joints, 3]
    """

    keypoints_3d = np.empty([num_joint, 3])
    keypoints_3d.fill(np.nan)
    num_cams = keypoints_mview.shape[0]
    # ic(num_cams)

    # cam_list = [i for i in range(num_cams)]
    # # ic(cam_list)
    # cam_combinations = list(itertools.combinations(cam_list, 2))
    # ic(cam_combinations)

    for j in range(num_joint):

        cams_detected = keypoints_mview[:, j, 2] > kpt_thr
        # ic(cams_detected)
        cam_idx = np.where(cams_detected)[0]
        # ic(cam_idx)
        if np.sum(cams_detected) < 2:
            continue
        cam_combinations = list(itertools.combinations(cam_idx, 2))
        # cam_combinations = get_all_combinations(cam_idx)

        # cam_set = set(range(keypoints_mview.shape[0]))
        inlier_set = set()
        # for i in range(niter):
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

            # ic(reprojection_error)
            new_inlier_set = set([i for i, v in enumerate(reprojection_error) if v < epsilon])
            # ic(reprojection_error)
            # ic(new_inlier_set)

            if len(new_inlier_set) > len(inlier_set):
                inlier_set = new_inlier_set
            if len(inlier_set) == num_cams:
                break
        if len(inlier_set) < 2:
            inlier_set = sampled_cam.copy()

        inlier_list = sorted(list(inlier_set))
        # ic(j, inlier_list)
        # if len(inlier_list) > 2:
            # ic(j, inlier_list)
        kp3d = triangulate(keypoints_mview[inlier_list, j, :2], projection_matrices[inlier_list])
        keypoints_3d[j, :] = kp3d

    return keypoints_3d


def compute_axis_lim(triangulated_points):
    # ic(triangulated_points.shape)
    # triangulated_points in shape [num_frame, num_keypoint, 3 axis]
    xlim, ylim, zlim = None, None, None
    # ic(triangulated_points.shape)
    minmax = np.nanpercentile(triangulated_points, q=[0, 100], axis=(0)).T
    # ic(minmax)
    minmax *= 1.
    minmax_range = (minmax[:, 1] - minmax[:, 0]).max() / 2
    if xlim is None:
        mid_x = np.mean(minmax[0])
        xlim = mid_x - minmax_range, mid_x + minmax_range
    if ylim is None:
        mid_y = np.mean(minmax[1])
        ylim = mid_y - minmax_range, mid_y + minmax_range
    if zlim is None:
        mid_z = np.mean(minmax[2])
        zlim = mid_z - minmax_range, mid_z + minmax_range
    return xlim, ylim, zlim


if __name__ == "__main__":
    cam_dict = {'cam0': 21334181,
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
                'cam19': 21334211
                }

    cam_file = "./camera.json"
    cam_param = json.load(open(cam_file))
    R = np.array(cam_param['cam0']['R']).reshape([3, 3])
    T = np.array(cam_param['cam0']['T'])
    # ic(R, T)
    # ic(T@R)

    """skeleton define"""
    # data_info_file = '../configs/_base_/datasets/coco_wholebody.py'
    # config = mmcv.Config.fromfile(data_info_file)
    # dataset_info = DatasetInfo(config._cfg_dict['dataset_info'])
    # ic(dataset_info.skeleton)
    links = [[15, 13],
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

    """read 2d results."""

    # problem cam: cam3
    used_cams = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6',
                 'cam7', 'cam8', 'cam9', 'cam10', 'cam11', 'cam12', 'cam13',
                 'cam14', 'cam15', 'cam16', 'cam17', 'cam18', 'cam19']

    # used_cams = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6',
    #              'cam7', 'cam14', 'cam15', 'cam16', 'cam17', 'cam18', 'cam19']

    # used_cams = ['cam0', 'cam1', 'cam2', 'cam3', 'cam4', 'cam5']

    # used_cams = ['cam18', 'cam19', 'cam0', 'cam1']

    # used_cams = ['cam0', 'cam1']

    # used_cams = ['cam4', 'cam5']

    # used_cams = ['cam18', 'cam19']

    # used_cams = ['cam14', 'cam15', 'cam16', 'cam17','cam18', 'cam19']

    # used_cams = ['cam8', 'cam9', 'cam10', 'cam11', 'cam12', 'cam13']

    # used_cams = ['cam0', 'cam11', 'cam2']
    # used_cams = ['cam0', 'cam1']  # Best
    # used_cams = ['cam0', 'cam1', 'cam2'] # Best
    # used_cams = ['cam3', 'cam5', 'cam7']
    # used_cams = ['cam3', 'cam4', 'cam8']

    # ic(proj_mat.shape)

    # define frame number
    # ff = 1
    xlim, ylim, zlim = None, None, None

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'../kp_3d/output.avi', fourcc, fps=30, frameSize=[1000, 1000])

    for ff in range(1, 687):
    # for ff in range(50,51):
        kp_2d_all_cams = []
        cam_ff = used_cams.copy()
        for cc in used_cams:
            try:
                file = f"../kp_2d/cello_0926_{cam_dict[cc]}/{ff}.json"
                kp_2d_cc_ff = np.array(json.load(open(file)))
                kp_2d_all_cams.append(kp_2d_cc_ff)
            except FileNotFoundError as e:
                # remove camera that drop frames
                cam_ff.remove(cc)
        # make projection matrix using filtered camera
        # ic(cam_ff)
        proj_mat = make_projection_matrix(cam_param, cams=cam_ff)
        kp_2d_all_cams = np.array(kp_2d_all_cams)
        # ic(kp_2d_all_cams)
        # kp_3d = triangulate_joints(kp_2d_all_cams, proj_mat, num_joint=133, kpt_thr=0.6)
        kp_3d = ransac_triangulate_joints(kp_2d_all_cams, proj_mat, num_joint=133, niter=20, epsilon=15, kpt_thr=0.5)
        # Remove hand and face
        # kp_2d_all_cams = kp_2d_all_cams[:, 0:24, :]
        # kp_3d = ransac_triangulate_joints(kp_2d_all_cams, proj_mat, num_joint=23, niter=20, epsilon=130)
        # ic(kp_3d.shape)
        if xlim is None:
            xlim, ylim, zlim = compute_axis_lim(kp_3d)
        # ic(xlim, ylim, zlim)
        fig = plt.figure(figsize=[10, 10])
        axes3 = fig.add_subplot(projection="3d")
        axes3.view_init(azim=-60, elev=30, roll=15)
        # view = (0, 90)
        axes3.set_xlim3d(xlim)
        axes3.set_ylim3d(ylim)
        axes3.set_zlim3d(zlim)
        axes3.set_box_aspect((1, 1, 1))
        axes3.scatter(kp_3d[:, 0],
                      kp_3d[:, 1],
                      kp_3d[:, 2], s=5)
        # ic(kp_3d)
        segs3d = kp_3d[tuple([links])]
        # ic(np.array(links).shape)
        # ic(segs3d.shape)
        coll_3d = Line3DCollection(segs3d, linewidths=1)
        axes3.add_collection(coll_3d)
        # plt.show()

        if not os.path.exists(f'../kp_3d/'):
            os.makedirs(f'../kp_3d/')

        plt.savefig(f'../kp_3d/sample{ff}.jpg')

        canvas = fig.canvas
        canvas.draw()
        width, height = canvas.get_width_height()
        image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_array = image_array.reshape(height, width, 3)
        image_array = image_array[:, :, ::-1]  # rgb to bgr
        out.write(image_array)
        plt.close()

    # ffmpeg -r 30 -i sample%d.jpg output.mp4 -crf 0
