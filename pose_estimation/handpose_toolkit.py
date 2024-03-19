import numpy as np
import os
import json
from icecream import ic
from scipy.spatial.transform import Rotation, Slerp
from triangulation.triangulation_pipeline import visualize_3d, CAM_DICT
import matplotlib.pyplot as plt
from triangulation.triangulation_pipeline import compute_axis_lim
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def get6d_from_txt(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    lh = np.array([float(x) for x in lines[3].rstrip().split(' ')]).reshape(16, 6)
    rh = np.array([float(x) for x in lines[1].rstrip().split(' ')]).reshape(16, 6)
    # ic(lh)
    f.close()
    return lh, rh


def is_orthogonal(matrix):
    transpose_matrix = np.transpose(matrix)
    product = np.dot(matrix, transpose_matrix)
    identity_matrix = np.identity(matrix.shape[0])
    return np.allclose(product, identity_matrix)


def rotation_6d_to_R(d6):
    a1, a2 = d6[:, :3], d6[:, 3:]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=1, keepdims=True)
    b3 = np.cross(b1, b2)
    b3 = b3 / np.linalg.norm(b3, axis=1, keepdims=True)
    R = np.stack((b1, b2, b3), axis=-2)

    return R


def get_mano_init(hand_type='left'):

    filepath = f"./mano_info/J3_{hand_type}.txt"
    with open(filepath) as f:
        lines = f.readlines()
    init_pose = np.zeros((21, 3))
    for l in range(len(lines)):
        init_pose[l] = np.array([float(x) for x in lines[l].rstrip().split(' ')])
    init_pose = init_pose - init_pose[0]

    return init_pose

def normalize_vector(v):
    magnitude = np.linalg.norm(v)
    if magnitude == 0:
        return v
    normalized_vector = v / magnitude
    return normalized_vector

def cal_dist(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2)))

# def get_joint_positions(positions, rotations, bone_lengths, parent_indices):
#
#     positions_original = positions.copy()
#     for i in range(1, 21):
#         parent_index = parent_indices[i]
#         parent_position = positions[parent_index]
#
#         R = rotations[parent_index]
#         while parent_index != 0:
#             parent_index = parent_indices[parent_index]
#             R = np.dot(rotations[parent_index], R)
#
#         bone_length = bone_lengths[i - 1]
#         original_parent_position = positions_original[parent_index]
#         original_self_position = positions_original[i]
#         original_vector = bone_length * normalize_vector(original_self_position - original_parent_position)
#
#         # 计算相对于父关节的位移
#         relative_position = np.dot(R, original_vector)
#
#         # 累加得到全局位置
#         positions[i] = relative_position + parent_position
#
#     return positions


def get_joint_positions(init_positions, rotations, bone_lengths, parent_indices):

    positions = init_positions.copy()
    for i in range(1, 21):
        parent_index = parent_indices[i]
        parent_position = positions[parent_index]

        R = rotations[parent_index]
        while parent_index != 0:
            parent_index = parent_indices[parent_index]
            R = np.dot(rotations[parent_index], R)

        bone_length = bone_lengths[i - 1]
        original_parent_position = init_positions[parent_index]
        original_self_position = init_positions[i]
        original_vector = bone_length * normalize_vector(original_self_position - original_parent_position)

        # 计算相对于父关节的位移
        relative_position = np.dot(R, original_vector)

        # 累加得到全局位置
        positions[i] = relative_position + parent_position

    return positions


def get_frame_info(proj_dir, frame_num):

    cam_dirs = os.listdir(proj_dir)
    frame_info = {}
    for cam_dir in cam_dirs:
        cam_num = cam_dir.split('_')[-1]
        filepath = os.path.join(proj_dir, cam_dir, f'{frame_num[cam_num]}.txt')

        lh, rh = get6d_from_txt(filepath)

        R_lh = rotation_6d_to_R(lh)
        R_rh = rotation_6d_to_R(rh)

        Q_lh = Rotation.from_matrix(R_lh).as_quat()
        Q_rh = Rotation.from_matrix(R_rh).as_quat()

        frame_info[f'{cam_num}'] = {'R_lh': R_lh,
                                    'R_rh': R_rh,
                                    'Q_lh': Q_lh,
                                    'Q_rh': Q_rh}

    return frame_info

def weighted_average_quaternion(q1, q2, q1_t, q2_t, w):
    key_rots = Rotation.from_quat((q1, q2))

    # 创建 Slerp 对象
    slerp = Slerp([q1_t, q2_t], key_rots)

    # 进行球面线性插值
    interpolated_quaternion = slerp(w)

    return interpolated_quaternion.as_quat()

def get_converted_R0(R0_cam, R0, cam_file_path):
    # cam_R_path = '../triangulation/jsons/cello_1113_scale_camera.json'

    with open(cam_file_path, 'r') as f:
        data_dict = json.load(f)

    cam_R = np.array(data_dict[R0_cam]['R']).reshape(3, 3)

    converted_R0 = np.dot(cam_R, R0)
    converted_R0 = converted_R0[np.newaxis, :]

    return converted_R0


def get_averaged_R(frame_info, R0_cam, cam_weights_lh, cam_weights_rh, cam_file_path):
    cams = list(cam_weights_lh.keys())

    averaged_Qs_lh = []
    averaged_Qs_rh = []

    for i in range(15):
        joint_index = i + 1

        q_current_lh = frame_info[cams[0]]['Q_lh'][joint_index]
        q_current_rh = frame_info[cams[0]]['Q_rh'][joint_index]

        weights_lh = cam_weights_lh[cams[0]]
        weights_rh = cam_weights_rh[cams[0]]

        for c in range(1, len(cams)):
            q_next_lh = frame_info[cams[c]]['Q_lh'][joint_index]
            q_next_rh = frame_info[cams[c]]['Q_rh'][joint_index]

            alpha_lh = cam_weights_lh[cams[c]] / (weights_lh + cam_weights_lh[cams[c]])
            alpha_rh = cam_weights_rh[cams[c]] / (weights_rh + cam_weights_rh[cams[c]])

            if not np.isnan(q_next_lh).any():
                q_current_lh = weighted_average_quaternion(q_current_lh, q_next_lh, 0, 1, alpha_lh)
            if not np.isnan(q_next_rh).any():
                q_current_rh = weighted_average_quaternion(q_current_rh, q_next_rh, 0, 1, alpha_rh)

            weights_lh += cam_weights_lh[cams[c]]
            weights_rh += cam_weights_rh[cams[c]]

        averaged_Qs_lh.append(q_current_lh)
        averaged_Qs_rh.append(q_current_rh)

    # 创建旋转对象，q 转 r
    R_matrix_lh = Rotation.from_quat(averaged_Qs_lh).as_matrix()
    R_matrix_rh = Rotation.from_quat(averaged_Qs_rh).as_matrix()


    # 直接用cam0-181的所有R参数, 注释掉则用averaged R
    #===========================================================
    # R_matrix_lh = []
    # R_matrix_rh = []
    # for i in range(15):
    #     joint_index = i + 1
    #     R_lh_181 = frame_info[cams[0]]['R_lh'][joint_index]
    #     R_rh_181 = frame_info[cams[0]]['R_rh'][joint_index]
    #     R_matrix_lh.append(R_lh_181)
    #     R_matrix_rh.append(R_rh_181)
    # R_matrix_lh = np.array(R_matrix_lh)
    # R_matrix_rh = np.array(R_matrix_rh)
    # ic(R_matrix_lh.shape)
    #===========================================================


    # 在最开始添加R0
    cam_num = str(CAM_DICT[R0_cam])

    R0_lh = frame_info[cam_num]['R_lh'][0]
    R0_lh_converted = get_converted_R0(R0_cam, R0_lh, cam_file_path)
    R_matrix_lh = np.vstack((R0_lh_converted, R_matrix_lh))

    R0_rh = frame_info[cam_num]['R_rh'][0]
    R0_rh_converted = get_converted_R0(R0_cam, R0_rh, cam_file_path)
    R_matrix_rh = np.vstack((R0_rh_converted, R_matrix_rh))


    return R_matrix_lh, R_matrix_rh


def visualize_hand(data, connections):
    fig = plt.figure(figsize=[10, 10])
    axes3 = fig.add_subplot(projection="3d", computed_zorder=False)
    xlim, ylim, zlim = compute_axis_lim(data)
    axes3.set_xlim3d(xlim)
    axes3.set_ylim3d(ylim)
    axes3.set_zlim3d(zlim)
    axes3.set_box_aspect((1, 1, 1))
    axes3.scatter(data[:, 0], data[:, 1], data[:, 2], s=50, zorder=1)

    hand_segs3d = data[tuple([connections])]
    left_hand_coll_3d = Line3DCollection(hand_segs3d, linewidths=1, zorder=1)
    axes3.add_collection(left_hand_coll_3d)
    axes3.view_init(elev=30, azim=45)
    plt.show()






