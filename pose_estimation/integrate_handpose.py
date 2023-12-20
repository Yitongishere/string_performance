import json
import os.path

import numpy as np
from icecream import ic
from handpose_toolkit import (get_mano_init, get_joint_positions,
                            cal_dist, get_averaged_R, visualize_hand,)
from triangulation.triangulation import visualize_3d


def get_bone_length_dw(kp_3d_all, frame_num):
    target_frame = kp_3d_all[frame_num-1]
    hands_dw = target_frame[LEFT_WRIST_INDEX:LEFT_WRIST_INDEX+42]
    hands_dw = hands_dw.reshape(2, 21, 3)
    hands_mano = hands_dw.copy()

    for dict_id, mano_id in enumerate(MANO_TO_DW):
        hands_mano[:, mano_id, :] = hands_dw[:, dict_id, :]

    bone_lengths = np.zeros([2, 20])
    for hand in range(2):  # two hands
        for idx, connection in enumerate(MANO_CONNECTIONS):
            point1 = hands_mano[hand][connection[0]]
            point2 = hands_mano[hand][connection[1]]
            bone_lengths[hand][idx] = cal_dist(point1, point2)

    return bone_lengths


def get_hands_joints(dir_6d, frame, bone_lengths):
    """输出dwpose顺序的手部关键点坐标"""
    lh_mano = get_mano_init('left')
    rh_mano = get_mano_init('right')

    lh_rot_averaged, rh_rot_averaged = get_averaged_R(dir_6d, frame, 'cam0', CAM_WEIGHTS_LH, CAM_WEIGHTS_RH)

    lh_bone_length = bone_lengths[0]
    rh_bone_length = bone_lengths[1]

    lh_joints_mano = get_joint_positions(lh_mano, lh_rot_averaged, lh_bone_length, MANO_PARENTS_INDICES)
    rh_joints_mano = get_joint_positions(rh_mano, rh_rot_averaged, rh_bone_length, MANO_PARENTS_INDICES)
    # visualize_hand(lh_joints_mano, MANO_CONNECTIONS)

    lh_joints_dw = np.zeros(lh_joints_mano.shape)
    rh_joints_dw = np.zeros(rh_joints_mano.shape)
    for dict_id, mano_id in enumerate(MANO_TO_DW):
        lh_joints_dw[dict_id, :] = lh_joints_mano[mano_id, :]
        rh_joints_dw[dict_id, :] = rh_joints_mano[mano_id, :]

    return lh_joints_dw, rh_joints_dw


CAM_WEIGHTS_LH = {
        '21334181': 0.30,
        '21334237': 0.10,
        '21334190': 0.15,
        '21334211': 0.10,
        '21334180': 0.10,
        '21334209': 0.05,
        '21334221': 0.05,
        '21334219': 0.05,
        '21334208': 0.025,
        '21334186': 0.025,
        '21334184': 0.025,
        '21334238': 0.025,
        '21293326': 0,
        '21293325': 0
    }

CAM_WEIGHTS_RH = {
        '21334181': 0.30,
        '21334237': 0.10,
        '21334190': 0.10,
        '21334211': 0.10,
        '21334180': 0.10,
        '21334209': 0.10,
        '21334221': 0.05,
        '21334219': 0.025,
        '21334208': 0.025,
        '21334186': 0.025,
        '21334184': 0,
        '21334238': 0.025,
        '21293326': 0.025,
        '21293325': 0.025
}


MANO_PARENTS_INDICES = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 3, 6, 9, 12, 15]
MANO_TO_DW = [0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5, 6, 17, 10, 11, 12, 19, 7, 8, 9, 18]

MANO_CONNECTIONS = [(0, 1), (1, 2), (2, 3),
                    (0, 4), (4, 5), (5, 6),
                    (0, 7), (7, 8), (8, 9),
                    (0, 10), (10, 11), (11, 12),
                    (0, 13), (13, 14), (14, 15),
                    (3, 16), (6, 17), (9, 18), (12, 19), (15, 20)]

DW_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (0, 9), (9, 10), (10, 11), (11, 12),
                    (0, 13), (13, 14), (14, 15), (15, 16),
                    (0, 17), (17, 18), (18, 19), (19, 20)]
LEFT_WRIST_INDEX = 91
RIGHT_WRIST_INDEX = 112


if __name__ == "__main__":
    proj_dir = "cello_1113_scale"
    dir_6d = f"./6d_result/{proj_dir}"

    with open(f'../kp_3d_result/{proj_dir}/kp_3d_all_dw.json', 'r') as f:
        data_dict = json.load(f)
    kp_3d_all = np.array(data_dict['kp_3d_all_dw'])

    bone_lengths = get_bone_length_dw(kp_3d_all, 1)

    for frame in range(len(kp_3d_all)):
        lh_wrist = kp_3d_all[frame][LEFT_WRIST_INDEX]
        rh_wrist = kp_3d_all[frame][RIGHT_WRIST_INDEX]

        lh_joints, rh_joints = get_hands_joints(dir_6d, frame+1, bone_lengths)

        if frame == 0:
            visualize_hand(lh_joints, DW_CONNECTIONS)

        lh_joints_pano = lh_joints + lh_wrist
        rh_joints_pano = rh_joints + rh_wrist

        kp_3d_all[frame][LEFT_WRIST_INDEX:LEFT_WRIST_INDEX+42] = np.vstack((lh_joints_pano, rh_joints_pano))
        print(f'Hand pose integrated for frame {frame+1}.')

    ic(kp_3d_all.shape)
    # visualize_3d(kp_3d_all, proj_dir, 'tri_3d_pe')

    if not os.path.exists(f'./{proj_dir}'):
        os.mkdir(f'./{proj_dir}')

    data_dict = {'kp_3d_all_pe': kp_3d_all.tolist()}
    with open(f'./{proj_dir}/kp_3d_all_pe.json', 'w') as f:
        json.dump(data_dict, f)