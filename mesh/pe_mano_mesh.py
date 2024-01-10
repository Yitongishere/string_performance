import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.spatial.transform import Rotation
from manopth.manolayer import ManoLayer
from manopth import demo
import open3d as o3d
import json
from icecream import ic
from pose_estimation.handpose_toolkit import get6d_from_txt, rotation_6d_to_R
from triangulation.triangulation import compute_axis_lim, HUMAN_LINKS, CELLO_LINKS, BOW_LINKS, LEFT_HAND_LINKS, \
    STRING_LINKS


def get_rot_vec(matrix):
    """
    Input: n * 3 * 3 ndarray
    Return: n * 3 ndarray
    """
    r = Rotation.from_matrix(matrix)
    vec = r.as_rotvec()
    return vec


# def visualize_mesh(data, vertices, triangles):
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
#         # mesh.set_edgecolor(edge_color)
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


if __name__ == '__main__':
    proj_dir = "cello_1113_scale"
    mesh_type = 'pe_mesh'
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

    with open(f'../pose_estimation/ik_result/{proj_dir}/hand_rot_vec.json', 'r') as f:
        data_dict = json.load(f)
    hand_rot_vec = np.array(data_dict['hand_rot_vec'])

    with open(f'../pose_estimation/ik_result/{proj_dir}/kp_3d_ik_smooth.json', 'r') as f:
        data_dict = json.load(f)
    kp_3d_ik_smooth = np.array(data_dict['kp_3d_ik_smooth'])

    for frame_id, kp_3d in enumerate(kp_3d_ik_smooth):

        wrist_target = kp_3d[91]

        hand_rot_vec_f = hand_rot_vec[frame_id].reshape(1, 48)

        # Initialize MANO layer
        mano_layer = ManoLayer(
            mano_root='mano/models', use_pca=False, flat_hand_mean=True, side='left')

        # Generate random shape parameters
        random_shape = torch.zeros(1, 10)
        # Generate random pose parameters, including 3 values for global axis-angle rotation
        # random_pose = torch.rand(1, ncomps + 3)
        pose = torch.tensor(hand_rot_vec_f, dtype=torch.float32)
        # ic(pose.shape)

        # Forward pass through MANO layer
        hand_verts, hand_joints = mano_layer(pose)
        hand_faces = mano_layer.th_faces

        hand_verts = hand_verts.numpy().squeeze() / 2000
        hand_joints = hand_joints.numpy().squeeze() / 2000

        wrist_origin = hand_joints[0]
        trans = wrist_target - wrist_origin
        # ic(trans)

        hand_verts = hand_verts + trans
        hand_joints = hand_joints + trans
        # ic(hand_verts)

        img_rgb = visualize_3d_mesh(kp_3d, hand_verts, hand_faces, view_angle=view)
        img_bgr = img_rgb[:, :, ::-1]  # rgb to bgr
        cv2.imwrite(f'./mesh_result/{proj_dir}/{mesh_type}/{frame_id}.png', img_bgr)
        out.write(img_bgr)

        print(f'Frame {frame_id} mesh generated.')
