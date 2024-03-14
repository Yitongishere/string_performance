import argparse
import json
import os.path

import numpy as np
import torch
from icecream import ic
from handpose_toolkit import (get_mano_init, get_joint_positions,
                              cal_dist, get_averaged_R, visualize_hand, )
from pose_estimation.manopth.manolayer import ManoLayer


def get_bone_length_dw(kp_3d_all, frame_num):
    target_frame = kp_3d_all[frame_num - 1]
    hands_dw = target_frame[LEFT_WRIST_INDEX:LEFT_WRIST_INDEX + 42]
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


def get_hand_length(hands_dw):
    hands_mano = hands_dw.copy()
    for dict_id, mano_id in enumerate(MANO_TO_DW):
        hands_mano[mano_id] = hands_dw[dict_id]

    bone_lengths = np.zeros([20])
    for idx, connection in enumerate(MANO_CONNECTIONS):
        point1 = hands_mano[connection[0]]
        point2 = hands_mano[connection[1]]
        bone_lengths[idx] = cal_dist(point1, point2)

    return bone_lengths


def get_hands_joints(dir_6d, frame, bone_lengths, cam_file_path, show_cam, instrument, parent_dir, proj_dir):
    """输出dwpose顺序的手部关键点坐标"""
    lh_mano = get_mano_init('left')
    rh_mano = get_mano_init('right')

    if instrument == 'cello':
        cam_weight_lh = CAM_WEIGHTS_LH_CELLO.copy()
        cam_weight_rh = CAM_WEIGHTS_RH_CELLO.copy()
    else:
        cam_weight_lh = CAM_WEIGHTS_LH_VIOLIN.copy()
        cam_weight_rh = CAM_WEIGHTS_RH_VIOLIN.copy()
    keys = list(CAM_WEIGHTS_LH_CELLO.keys())
    values = list(CAM_WEIGHTS_LH_CELLO.values())
    for key in keys:
        try:
            frame_drop_path = f'../data/{parent_dir}/{proj_dir}/videos/{key}_FrameDropIDLog.txt'
            drop_frames = np.array(open(frame_drop_path).readlines(), dtype=int)
            if frame in drop_frames:
                print(f'Remove cam {key} for frame {frame}!')
                values.remove(cam_weight_lh[key])
                cam_weight_lh[key] = 0
                for k, v in cam_weight_lh.items():
                    cam_weight_lh[k] = v / sum(values)
                values = list(cam_weight_lh.values())
        except FileNotFoundError as e:
            pass

    cam_weight_lh = dict(sorted(cam_weight_lh.items(), key=lambda x: x[1], reverse=True))

    lh_rot_averaged, rh_rot_averaged = get_averaged_R(dir_6d, frame, show_cam, cam_weight_lh, cam_weight_rh,
                                                      cam_file_path)

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

    return lh_joints_dw, rh_joints_dw, lh_rot_averaged, rh_rot_averaged


def get_lh_bone_length(proj_dir):
    with open(f'../audio/cp_result/{proj_dir}/kp_3d_all_dw_cp_smooth.json', 'r') as f:
        data_dict = json.load(f)
    kp_3d_dw = np.array(data_dict['kp_3d_all_dw_cp_smooth'])

    kp_3d = kp_3d_dw[0]

    # hand_joints_dw = kp_3d[91:112]
    hand_joints_dw_rh = kp_3d[112:133]

    # Initialize MANO layer
    mano_layer = ManoLayer(mano_root='./mano/models', use_pca=False, flat_hand_mean=True, side='right')
    pose_0 = np.zeros([1, 48])
    # pose_0 = np.zeros([16, 3])
    # pose_0[0] = global_rot
    # pose_0 = pose_0.reshape(1, 48)
    pose_0 = torch.tensor(pose_0, dtype=torch.float32)
    # Forward pass through MANO layer
    _, hand_joints_rh_0 = mano_layer(pose_0)
    hand_joints_rh_0 = hand_joints_rh_0.numpy().squeeze()

    bone_length_mano_rh = get_hand_length(hand_joints_rh_0)
    # ic(bone_length_mano_rh)

    mano_layer = ManoLayer(mano_root='./mano/models', use_pca=False, flat_hand_mean=True, side='left')
    _, hand_joints_lh_0 = mano_layer(pose_0)
    hand_joints_lh_0 = hand_joints_lh_0.numpy().squeeze()

    bone_length_mano_lh = get_hand_length(hand_joints_lh_0)
    # ic(bone_length_mano_lh)

    bone_length_dw_rh = get_hand_length(hand_joints_dw_rh)
    # ic(bone_length_dw_rh)

    ratio = np.average(bone_length_mano_rh / bone_length_dw_rh)
    # ic(ratio)

    bone_length_mano_lh_scale = bone_length_mano_lh / ratio
    bone_length_mano_lh_scale = bone_length_mano_lh_scale.tolist()
    # ic(bone_length_mano_lh_scale)

    return bone_length_mano_lh_scale


CAM_WEIGHTS_LH_VIOLIN = {
    # 7
    '21334207': 0.4,
    '21334191': 0.1,
    '21334182': 0.05,
    # 6
    '21334220': 0.4,  # TODO reduce weight
    # '21334236': 0,  # TODO add weight
    '21334183': 0.05,
    # 5
    '21293324': 0,
    '21334218': 0,
    '21334206': 0
}

CAM_WEIGHTS_RH_VIOLIN = {
    # 7
    '21334207': 0.2,
    '21334191': 0.1,
    '21334182': 0.05,
    # 6
    '21334220': 0.6,  # TODO reduce weight
    # '21334236': 0,  # TODO add weight
    '21334183': 0.05,
    # 5
    '21293324': 0,
    '21334218': 0,
    '21334206': 0
}

CAM_WEIGHTS_LH_CELLO = {
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

CAM_WEIGHTS_RH_CELLO = {
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
    parser = argparse.ArgumentParser(prog='integrate_handpose_pipeline')
    parser.add_argument('--cam_file', default='../triangulation/jsons/cello_1113_scale_camera.json', type=str,
                        required=True)
    parser.add_argument('--parent_dir', default='cello_1113', type=str, required=True)
    parser.add_argument('--proj_dir', default='cello_1113_scale', type=str, required=True)
    parser.add_argument('--start_frame', default=128, type=str, required=True)
    parser.add_argument('--instrument', default='cello', type=str, required=True)
    parser.add_argument('--cam_num', default='cam0', type=str, required=True)
    args = parser.parse_args()
    cam_file = args.cam_file
    parent_dir = args.parent_dir
    proj_dir = args.proj_dir
    start_frame = int(args.start_frame)
    instrument = args.instrument
    cam_num = args.cam_num

    # proj_dir = "cello_1113_scale"
    # start_frame = 128
    # cam_file = '../triangulation/jsons/cello_1113_scale_camera.json'

    dir_6d = f"./6d_result/{proj_dir}"

    lh_bone_length = get_lh_bone_length(proj_dir)

    # with open(f'../kp_3d_result/{proj_dir}/kp_3d_all_dw.json', 'r') as f:
    #     data_dict = json.load(f)
    # kp_3d_all = np.array(data_dict['kp_3d_all_dw'])

    with open(f'../audio/cp_result/{proj_dir}/kp_3d_all_dw_cp.json', 'r') as f:
        data_dict = json.load(f)
    kp_3d_all = np.array(data_dict['kp_3d_all_dw_cp'])

    # bone_lengths = get_bone_length_dw(kp_3d_all, 1)
    # print(bone_lengths)
    rh_bone_length = get_bone_length_dw(kp_3d_all, 1)[1]
    bone_lengths = np.array([lh_bone_length, rh_bone_length])

    integrated_hand_rot = []
    for frame in range(len(kp_3d_all)):
        lh_wrist = kp_3d_all[frame][LEFT_WRIST_INDEX]
        rh_wrist = kp_3d_all[frame][RIGHT_WRIST_INDEX]

        lh_joints, rh_joints, lh_rot, rh_rot = get_hands_joints(dir_6d, frame + start_frame, bone_lengths, cam_file,
                                                                cam_num, instrument, parent_dir, proj_dir)

        hand_rot = np.concatenate([lh_rot, rh_rot], axis=0)
        integrated_hand_rot.append(hand_rot.tolist())

        # if frame == 0:
        #     visualize_hand(lh_joints, DW_CONNECTIONS)

        lh_joints_pano = lh_joints + lh_wrist
        rh_joints_pano = rh_joints + rh_wrist

        # kp_3d_all[frame][LEFT_WRIST_INDEX:LEFT_WRIST_INDEX+42] = np.vstack((lh_joints_pano, rh_joints_pano))
        kp_3d_all[frame][LEFT_WRIST_INDEX:RIGHT_WRIST_INDEX] = lh_joints_pano
        print(f'Hand pose integrated for frame {frame + start_frame}.')

    ic(kp_3d_all.shape)
    # visualize_3d(kp_3d_all, proj_dir, 'integrated_fk_3d_pe')

    # if not os.path.exists(f'../kp_3d_result/{proj_dir}'):
    #     os.mkdir(f'../kp_3d_result/{proj_dir}')
    #
    # data_dict = {'kp_3d_all_pe': kp_3d_all.tolist()}
    # with open(f'../kp_3d_result/{proj_dir}/kp_3d_all_pe.json', 'w') as f:
    #     json.dump(data_dict, f)

    if not os.path.exists(f'./fk_result/{proj_dir}'):
        os.makedirs(f'./fk_result/{proj_dir}')

    data_dict = {'kp_3d_all_pe_fk': kp_3d_all.tolist()}
    with open(f'./fk_result/{proj_dir}/kp_3d_all_pe_fk.json', 'w') as f:
        json.dump(data_dict, f)

    data_dict = {'integrated_hand_rot': integrated_hand_rot}
    with open(f'./fk_result/{proj_dir}/integrated_hand_rot.json', 'w') as f:
        json.dump(data_dict, f)
