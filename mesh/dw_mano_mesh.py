import json

import cv2
import numpy as np
import torch
from icecream import ic
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.spatial.transform import Rotation
from triangulation.triangulation import compute_axis_lim, STRING_LINKS, HUMAN_LINKS, CELLO_LINKS, BOW_LINKS, \
    LEFT_HAND_LINKS
from mesh.minimal_ik.armatures import *
from mesh.minimal_ik.models import *
from mesh.manopth.manolayer import ManoLayer
from mesh.minimal_ik import config
from pose_estimation.handpose_toolkit import get_mano_init
from triangulation.triangulation import compute_axis_lim

MANO_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 16),
                    (0, 4), (4, 5), (5, 6), (6, 17),
                    (0, 7), (7, 8), (8, 9), (9, 18),
                    (0, 10), (10, 11), (11, 12), (12, 19),
                    (0, 13), (13, 14), (14, 15), (15, 20)]

DW_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4),
                  (0, 5), (5, 6), (6, 7), (7, 8),
                  (0, 9), (9, 10), (10, 11), (11, 12),
                  (0, 13), (13, 14), (14, 15), (15, 16),
                  (0, 17), (17, 18), (18, 19), (19, 20)]

DW_PARENTS_INDICES = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]

DW_CHILD_INDICES = [1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]

ROT_DW_TO_MANO = [0, 4, 5, 6, 7, 8, 9, 13, 14, 15, 10, 11, 12, 1, 2, 3]

DW_ROT_TO_POS = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]


# def visualize_3d_mesh(data, vertices, triangles):
#     xlim, ylim, zlim = None, None, None
#     framenum = data.shape[0]
#     key_points_num = data.shape[1]
#
#     # if not os.path.exists(f'../kp_3d_result/'):
#     #     os.makedirs(f'../kp_3d_result/')
#     #
#     # if not os.path.exists(f'../kp_3d_result/{proj_path}/'):
#     #     os.makedirs(f'../kp_3d_result/{proj_path}/')
#     #
#     # if not os.path.exists(f'../kp_3d_result/{proj_path}/{file_path}'):
#     #     os.makedirs(f'../kp_3d_result/{proj_path}/{file_path}')
#     #
#     # fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     # out = cv2.VideoWriter(f'../kp_3d_result/{proj_path}/{file_path}/output_{view_angle}.avi', fourcc, fps=30,
#     #                       frameSize=[1000, 1000])
#
#     for f in range(framenum):
#         kp_3d = data[f]
#         fig = plt.figure(figsize=[10, 10])
#         axes3 = fig.add_subplot(projection="3d", computed_zorder=False)
#         if xlim is None:
#             xlim, ylim, zlim = compute_axis_lim(data[f])
#
#         axes3.view_init(azim=135, elev=-20, roll=-45)
#
#         print(xlim)
#         axes3.set_xlim3d(xlim)
#         axes3.set_ylim3d(ylim)
#         axes3.set_zlim3d(zlim)
#         axes3.set_box_aspect((1, 1, 1))
#
#         human_segs3d = kp_3d[tuple([HUMAN_LINKS])]
#         cello_segs3d = kp_3d[tuple([CELLO_LINKS])]
#         bow_segs3d = kp_3d[tuple([BOW_LINKS])]
#
#         human_coll_3d = Line3DCollection(human_segs3d, linewidths=1, zorder=1)
#         cello_coll_3d = Line3DCollection(cello_segs3d, edgecolors='saddlebrown', linewidths=1, zorder=1)
#         bow_coll_3d = Line3DCollection(bow_segs3d, edgecolors='goldenrod', linewidths=1, zorder=1)
#
#         axes3.add_collection(human_coll_3d)
#         axes3.add_collection(cello_coll_3d)
#         axes3.add_collection(bow_coll_3d)
#
#         axes3.scatter(kp_3d[0:133, 0],
#                       kp_3d[0:133, 1],
#                       kp_3d[0:133, 2], s=5, zorder=1)
#         axes3.scatter(kp_3d[133:140, 0],
#                       kp_3d[133:140, 1],
#                       kp_3d[133:140, 2], c='saddlebrown', s=5, zorder=1)
#         axes3.scatter(kp_3d[140:142, 0],
#                       kp_3d[140:142, 1],
#                       kp_3d[140:142, 2], c='goldenrod', s=5, zorder=1)
#         vertices_f = vertices[f]
#         # axes3.scatter(vertices_f[:, 0], vertices_f[:, 1], vertices_f[:, 2], alpha=0.1)
#         mesh = Poly3DCollection(vertices_f[triangles], alpha=0.2)
#         # face_color = (141 / 255, 184 / 255, 226 / 255)
#         # edge_color = (50 / 255, 50 / 255, 50 / 255)
#         face_color = (128 / 255, 128 / 255, 128 / 255)
#         edge_color = (128 / 255, 128 / 255, 128 / 255)
#         mesh.set_edgecolor(edge_color)
#         mesh.set_facecolor(face_color)
#         axes3.add_collection3d(mesh)
#         if key_points_num > 142:
#             axes3.scatter(kp_3d[142:150, 0],
#                           kp_3d[142:150, 1],
#                           kp_3d[142:150, 2], c='black', s=5, zorder=98)
#             string_segs3d = kp_3d[tuple([STRING_LINKS])]
#             string_coll_3d = Line3DCollection(string_segs3d, edgecolors='black', linewidths=1, zorder=98)
#             axes3.add_collection(string_coll_3d)
#             if True not in np.isnan(kp_3d[150]):
#                 axes3.scatter(kp_3d[150, 0],
#                               kp_3d[150, 1],
#                               kp_3d[150, 2], c='r', s=5,
#                               zorder=100)  # zorder must be the biggest so that it would not be occluded
#                 # axes3.scatter(kp_3d[151:155, 0],
#                 #               kp_3d[151:155, 1],
#                 #               kp_3d[151:155, 2], c='orange', s=5, zorder=99)
#
#         plt.show()


def visualize_3d_mesh(data, vertices, triangles, view_angle='whole'):
    xlim, ylim, zlim = None, None, None
    key_points_num = data.shape[0]

    zoom_in = view_angle == 'finger'
    fig = plt.figure(figsize=[10, 10])
    axes3 = fig.add_subplot(projection="3d", computed_zorder=False)
    if xlim is None:
        if zoom_in:
            string_data = data[142:155]
            left_arm_data = data[5:10:2]
            string_with_arm = np.concatenate((string_data, left_arm_data), axis=0)
            xlim, ylim, zlim = compute_axis_lim(string_with_arm, scale_factor=1.5)
        else:
            xlim, ylim, zlim = compute_axis_lim(data)

    if zoom_in:
        axes3.view_init(azim=90, elev=30, roll=-45)
    else:
        axes3.view_init(azim=135, elev=-20, roll=-45)
    # ic(xlim, ylim, zlim)
    # axes3.set_box_aspect(aspect=[5, 5, 5])

    axes3.set_xlim3d(xlim)
    axes3.set_ylim3d(ylim)
    axes3.set_zlim3d(zlim)
    axes3.set_box_aspect((1, 1, 1))

    human_segs3d = data[tuple([HUMAN_LINKS])]
    cello_segs3d = data[tuple([CELLO_LINKS])]
    bow_segs3d = data[tuple([BOW_LINKS])]
    left_hand_segs3d = data[tuple([LEFT_HAND_LINKS])]

    if zoom_in:
        if key_points_num > 142:
            string_segs3d = data[tuple([STRING_LINKS])]
            string_coll_3d = Line3DCollection(string_segs3d, edgecolors='black', linewidths=1, zorder=98)
            axes3.add_collection(string_coll_3d)
        # left_hand_coll_3d = Line3DCollection(left_hand_segs3d, linewidths=1, zorder=1)
        # axes3.add_collection(left_hand_coll_3d)
        # axes3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], alpha=0.1)
        mesh = Poly3DCollection(vertices[triangles], alpha=0.2)
        # face_color = (141 / 255, 184 / 255, 226 / 255)
        # edge_color = (50 / 255, 50 / 255, 50 / 255)
        face_color = (128 / 255, 128 / 255, 128 / 255)
        edge_color = (128 / 255, 128 / 255, 128 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        axes3.add_collection3d(mesh)
        # # left hand
        # axes3.scatter(data[91:112, 0],
        #               data[91:112, 1],
        #               data[91:112, 2], s=5, c='#1f77b4', zorder=1)
        # # left arm
        # axes3.scatter(data[5:10:2, 0],
        #               data[5:10:2, 1],
        #               data[5:10:2, 2], s=5, c='#1f77b4', zorder=1)
        if key_points_num > 142:
            axes3.scatter(data[142:150, 0],
                          data[142:150, 1],
                          data[142:150, 2], c='black', s=5, zorder=98)
        if True not in np.isnan(data[150]):
            axes3.scatter(data[150, 0],
                          data[150, 1],
                          data[150, 2], c='r', s=30,
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

        axes3.scatter(data[0:133, 0],
                      data[0:133, 1],
                      data[0:133, 2], s=5, zorder=1)
        axes3.scatter(data[133:140, 0],
                      data[133:140, 1],
                      data[133:140, 2], c='saddlebrown', s=5, zorder=1)
        axes3.scatter(data[140:142, 0],
                      data[140:142, 1],
                      data[140:142, 2], c='goldenrod', s=5, zorder=1)
        # axes3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], alpha=0.1)
        mesh = Poly3DCollection(vertices[triangles], alpha=0.2)
        # face_color = (141 / 255, 184 / 255, 226 / 255)
        # edge_color = (50 / 255, 50 / 255, 50 / 255)
        face_color = (128 / 255, 128 / 255, 128 / 255)
        edge_color = (128 / 255, 128 / 255, 128 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        axes3.add_collection3d(mesh)
        if key_points_num > 142:
            axes3.scatter(data[142:150, 0],
                          data[142:150, 1],
                          data[142:150, 2], c='black', s=5, zorder=98)
            string_segs3d = data[tuple([STRING_LINKS])]
            string_coll_3d = Line3DCollection(string_segs3d, edgecolors='black', linewidths=1, zorder=98)
            axes3.add_collection(string_coll_3d)
            if True not in np.isnan(data[150]):
                axes3.scatter(data[150, 0],
                              data[150, 1],
                              data[150, 2], c='r', s=5,
                              zorder=100)  # zorder must be the biggest so that it would not be occluded
                axes3.scatter(data[151:155, 0],
                              data[151:155, 1],
                              data[151:155, 2], c='orange', s=5, zorder=99)

    # plt.show()

    canvas = fig.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(height, width, 3)
    plt.close()

    return image_array


def get_rot_vec(matrix):
    """
    Input: n * 3 * 3 ndarray
    Return: n * 3 ndarray
    """
    r = Rotation.from_matrix(matrix)
    vec = r.as_rotvec()
    return vec


def get_rot_mat(vector):
    """
        Input: n * 3 ndarray
        Return: n * 3 * 3 ndarray
        """
    r = Rotation.from_rotvec(vector)
    mat = r.as_matrix()
    return mat


def get_axis_ang(origin_vec, target_vec):
    origin_vec_norm = origin_vec / np.linalg.norm(origin_vec)
    target_vec_norm = target_vec / np.linalg.norm(target_vec)
    if np.array_equal(origin_vec_norm, target_vec_norm) or np.array_equal(origin_vec_norm, -target_vec_norm):
        return [0, 0, 0]
    else:
        theta = np.arccos(origin_vec_norm @ target_vec_norm)
        n_vec = np.cross(origin_vec_norm, target_vec_norm)
        n_vec = n_vec / np.linalg.norm(n_vec)
        return theta * n_vec


def visualize(data):
    xlim, ylim, zlim = None, None, None
    fig = plt.figure(figsize=[10, 10])
    axes3 = fig.add_subplot(projection="3d", computed_zorder=False)
    if xlim is None:
        xlim, ylim, zlim = compute_axis_lim(data)

    axes3.view_init(azim=135, elev=-20, roll=-45)

    axes3.set_xlim3d(xlim)
    axes3.set_ylim3d(ylim)
    axes3.set_zlim3d(zlim)
    axes3.set_box_aspect((1, 1, 1))
    axes3.scatter(data[:, 0], data[:, 1], data[:, 2], color='r')

    color = ['#fcb1b1', '#f0f696', '#96f7d2', '#f5b5fc', '#ff5959']
    for i in range(len(color)):
        hand_segs3d = data[tuple([DW_CONNECTIONS])][4 * i:4 * (i + 1), :]
        left_hand_coll_3d = Line3DCollection(hand_segs3d, linewidths=1, zorder=1, color=color[i])
        axes3.add_collection(left_hand_coll_3d)
    plt.show()


def get_joint_rot(init_pos, target_pos, wrist_rot, parent_indices, child_indices):
    joint_rot = np.zeros([16, 3])
    joint_rot[0] = wrist_rot
    # ic(init_pos.shape)
    # ic(target_pos.shape)
    for i in range(1, 16):
        parent_index = parent_indices[i]
        child_index = child_indices[i]
        cur_pos_idx = DW_ROT_TO_POS[i]
        init_child_pos = init_pos[child_index]
        init_cur_pos = init_pos[cur_pos_idx]
        init_vec = init_child_pos - init_cur_pos
        target_child_pos = target_pos[child_index]
        target_cur_pos = target_pos[cur_pos_idx]
        target_vec = target_child_pos - target_cur_pos
        total_rot_vec = get_axis_ang(init_vec, target_vec)
        total_rot_mat = get_rot_mat(total_rot_vec)
        accum_rot_mat = get_rot_mat(joint_rot[parent_index])
        while parent_index != 0:
            parent_index = parent_indices[parent_index]
            parent_rot_mat = get_rot_mat(joint_rot[parent_index])
            accum_rot_mat = np.matmul(parent_rot_mat, accum_rot_mat)
        cur_rot_mat = np.matmul(np.linalg.inv(accum_rot_mat), total_rot_mat)
        cur_rot_vec = get_rot_vec(cur_rot_mat)
        joint_rot[i] = cur_rot_vec
    return joint_rot


if __name__ == '__main__':

    proj_dir = "cello_1113_scale"
    mesh_type = 'dw_mesh'
    view = 'finger'

    if not os.path.exists(f'./mesh_result/'):
        os.makedirs(f'./mesh_result/')

    if not os.path.exists(f'./mesh_result/{proj_dir}'):
        os.makedirs(f'./mesh_result/{proj_dir}')

    if not os.path.exists(f'./mesh_result/{proj_dir}/{mesh_type}'):
        os.makedirs(f'./mesh_result/{proj_dir}/{mesh_type}')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'./mesh_result/{proj_dir}/{mesh_type}/output_{mesh_type}_{view}.avi', fourcc, fps=30,
                          frameSize=[1000, 1000])

    with open(f'../audio/{proj_dir}/kp_3d_all_dw_cp_smooth.json', 'r') as f:
        data_dict = json.load(f)
    kp_3d_dw = np.array(data_dict['kp_3d_all_dw_cp_smooth'])

    with open(f'../pose_estimation/ik_result/{proj_dir}/hand_rot_vec.json', 'r') as f:
        data_dict = json.load(f)
    hand_rot_vec = np.array(data_dict['hand_rot_vec'])

    for frame_id, kp_3d in enumerate(kp_3d_dw):
        global_rot = hand_rot_vec[frame_id][0]

        hand_joints_target = kp_3d[91:112]

        # Initialize MANO layer
        mano_layer = ManoLayer(mano_root='./mano/models', use_pca=False, flat_hand_mean=True, side='left')
        pose_0 = np.zeros([1, 48])
        # pose_0 = np.zeros([16, 3])
        # pose_0[0] = global_rot
        # pose_0 = pose_0.reshape(1, 48)
        pose_0 = torch.tensor(pose_0, dtype=torch.float32)
        # Forward pass through MANO layer
        _, hand_joints_0 = mano_layer(pose_0)
        hand_joints_0 = hand_joints_0.numpy().squeeze() / 2000
        # ic(hand_joints_0)
        # ic(hand_joints_target)
        trans = hand_joints_target[0] - hand_joints_0[0]
        hand_joints_0 = hand_joints_0 + trans
        # ic(hand_joints_0)

        # kp_3d_test = kp_3d.copy()
        # kp_3d_test[91:112] = hand_joints_0
        # kp_3d_test = kp_3d_test.reshape(1, 155, 3)
        # print(kp_3d_test)
        # visualize_mesh(kp_3d_test, None, None)

        est_rot_dw = get_joint_rot(hand_joints_0, hand_joints_target, global_rot, DW_PARENTS_INDICES, DW_CHILD_INDICES)
        # print(est_rot_dw)

        est_rot_mano = est_rot_dw.copy()

        for dict_id, dw_id in enumerate(ROT_DW_TO_MANO):
            est_rot_mano[dict_id, :] = est_rot_dw[dw_id, :]
            est_rot_mano[dict_id, :] = est_rot_dw[dw_id, :]

        est_rot_mano = est_rot_mano.reshape(1, 48)
        est_rot_mano = torch.tensor(est_rot_mano, dtype=torch.float32)

        hand_verts_est, hand_joints_est = mano_layer(est_rot_mano)
        hand_verts_est = hand_verts_est.numpy().squeeze() / 2000
        hand_joints_est = hand_joints_est.numpy().squeeze() / 2000
        hand_faces_est = mano_layer.th_faces

        hand_verts_est = hand_verts_est + trans
        hand_joints_est = hand_joints_est + trans

        # ic(hand_joints_est)
        # visualize(hand_joints_est)
        img_rgb = visualize_3d_mesh(kp_3d, hand_verts_est, hand_faces_est, view_angle=view)
        img_bgr = img_rgb[:, :, ::-1]  # rgb to bgr
        cv2.imwrite(f'./mesh_result/{proj_dir}/{mesh_type}/{frame_id}.png', img_bgr)
        out.write(img_bgr)

        print(f'Frame {frame_id} mesh generated.')
