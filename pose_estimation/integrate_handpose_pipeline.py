import argparse
import json
import os.path

import numpy as np
import torch
from icecream import ic
from handpose_toolkit import (get_mano_init, get_joint_positions,
                              cal_dist, get_averaged_R, visualize_hand, get_frame_info, )
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


def get_hands_joints(dir_6d, frame, bone_lengths, cam_param, show_cam, instrument, parent_dir, proj_dir, cam_drop_frames = None, filename_appendix=''):
    """输出dwpose顺序的手部关键点坐标"""
    lh_mano = get_mano_init('left', filename_appendix)
    rh_mano = get_mano_init('right', filename_appendix)

    if instrument == 'cello':
        cam_weight_lh = CAM_WEIGHTS_LH_CELLO.copy()
        cam_weight_rh = CAM_WEIGHTS_RH_CELLO.copy()
    else:
        cam_weight_lh = CAM_WEIGHTS_LH_VIOLIN.copy()
        cam_weight_rh = CAM_WEIGHTS_RH_VIOLIN.copy()

    dirs = os.listdir(dir_6d)
    run_cams = [i.split('_')[-1] for i in dirs]

    cam_weight_lh = {run_cam: cam_weight_lh[run_cam] for run_cam in run_cams if run_cam in cam_weight_lh}
    cam_weight_rh = {run_cam: cam_weight_rh[run_cam] for run_cam in run_cams if run_cam in cam_weight_rh}

    keys = list(cam_weight_lh.keys())
    values = list(cam_weight_lh.values())

    for cam_weight in [cam_weight_lh, cam_weight_rh]:
        keys = list(cam_weight.keys())
        values = list(cam_weight.values())
        frame_num_dict = {}
        for key in keys:
            actual_frame = frame
            frame_num_dict[f'{key}'] = actual_frame
            try:       
                if cam_drop_frames is not None:
                    drop_frames = [] if cam_drop_frames[key] == None else np.array(cam_drop_frames[key],dtype = int)
                
                dropped = 0
                for drop_frame in drop_frames:
                    if drop_frame < frame:
                        dropped += 1
                    elif drop_frame == frame:
                        print(f'Remove cam {key} for frame {frame}!')
                        values.remove(cam_weight[key])
                        cam_weight[key] = 0
                        for k, v in cam_weight.items():
                            cam_weight[k] = v / sum(values)
                        values = list(cam_weight.values())
                        break
                    elif drop_frame > frame:
                        break
                actual_frame -= dropped
                frame_num_dict[f'{key}'] = actual_frame
                # if frame in drop_frames:
                #     print(f'Remove cam {key} for frame {frame}!')
                #     values.remove(cam_weight_lh[key])
                #     cam_weight_lh[key] = 0
                #     for k, v in cam_weight_lh.items():
                #         cam_weight_lh[k] = v / sum(values)
                #     values = list(cam_weight_lh.values())
            except FileNotFoundError as e:
                pass

    cam_weight_lh = dict(sorted(cam_weight_lh.items(), key=lambda x: x[1], reverse=True))
    cam_weight_rh = dict(sorted(cam_weight_rh.items(), key=lambda x: x[1], reverse=True))


    frame_info = get_frame_info(dir_6d, frame_num_dict)


    lh_rot_averaged, rh_rot_averaged = get_averaged_R(frame_info,
                                                      show_cam,
                                                      cam_weight_lh,
                                                      cam_weight_rh,
                                                      cam_param)
    
    lh_bone_length = bone_lengths[0]
    rh_bone_length = bone_lengths[1]

    lh_joints_mano = get_joint_positions(lh_mano, lh_rot_averaged, MANO_PARENTS_INDICES, lh_bone_length)
    rh_joints_mano = get_joint_positions(rh_mano, rh_rot_averaged, MANO_PARENTS_INDICES, rh_bone_length)
    # visualize_hand(lh_joints_mano, MANO_CONNECTIONS)
    
    lh_joints_dw = np.zeros(lh_joints_mano.shape)
    rh_joints_dw = np.zeros(rh_joints_mano.shape)
    for dict_id, mano_id in enumerate(MANO_TO_DW):
        lh_joints_dw[dict_id, :] = lh_joints_mano[mano_id, :]
        rh_joints_dw[dict_id, :] = rh_joints_mano[mano_id, :]
    
    return lh_joints_dw, rh_joints_dw, lh_rot_averaged, rh_rot_averaged


def get_lh_bone_length(proj_dir):
    print(os.path.abspath(f'../audio/cp_result/{parent_dir}/{proj_dir}/kp_3d_all_dw_cp_smooth.json'))
    with open(f'../audio/cp_result/{parent_dir}/{proj_dir}/kp_3d_all_dw_cp_smooth.json', 'r') as f:
        data_dict = json.load(f)
    kp_3d_dw = np.array(data_dict['kp_3d_all_dw_cp_smooth'])

    hand_joints_dw_lh = kp_3d_dw[:, 91:112, :]

    # Initialize MANO layer
    mano_layer = ManoLayer(mano_root='./mano/models', use_pca=False, flat_hand_mean=True, side='right')
    pose_0 = np.zeros([1, 48])
    pose_0 = torch.tensor(pose_0, dtype=torch.float32)
    # Forward pass through MANO layer
    _, hand_joints_rh_0 = mano_layer(pose_0)
    hand_joints_rh_0 = hand_joints_rh_0.numpy().squeeze()

    bone_length_mano_rh = get_hand_length(hand_joints_rh_0)

    mano_layer = ManoLayer(mano_root='./mano/models', use_pca=False, flat_hand_mean=True, side='left')
    _, hand_joints_lh_0 = mano_layer(pose_0)
    
    hand_joints_lh_0 = hand_joints_lh_0.numpy().squeeze()

    bone_length_mano_lh = get_hand_length(hand_joints_lh_0)
    # bone_length_dw_rh = get_hand_length(hand_joints_dw_rh)

    bone_length_dw_lh_frame = np.array([get_hand_length(i) for i in hand_joints_dw_lh if not np.isnan(i).any()])

    bone_length_dw_lh_frameavg = np.average(bone_length_dw_lh_frame)
    ratio = np.average(bone_length_mano_rh / bone_length_dw_lh_frameavg)
    bone_length_mano_lh_scale = bone_length_mano_lh / ratio
    bone_length_mano_lh_scale = bone_length_mano_lh_scale.tolist()

    return bone_length_mano_lh_scale


CAM_WEIGHTS_LH_VIOLIN = {
    # 7
    '21334207': 0.3,
    '21334191': 0.,
    '21334182': 0.,
    # 6
    '21334220': 0.3,  
    '21334236': 0.3,  
    '21334183': 0.,
    # 5
    '21293324': 0.1,
    '21334218': 0,
    '21334206': 0
}

CAM_WEIGHTS_RH_VIOLIN = {
    # 7
    '21334207': 0.3,
    '21334191': 0.,
    '21334182': 0.,
    # 6
    '21334220': 0.3,
    '21334236': 0.3,
    '21334183': 0.,
    # 5
    '21293324': 0.1,
    '21334218': 0,
    '21334206': 0
}

# CAM_WEIGHTS_LH_CELLO = {
#     '21334181': 0.30,
#     '21334237': 0.10,
#     '21334190': 0.15,
#     '21334211': 0.10,
#     '21334180': 0.10,
#     '21334209': 0.05,
#     '21334221': 0.05,
#     '21334219': 0.05,
#     '21334208': 0.025,
#     '21334186': 0.025,
#     '21334184': 0.025,  # TODO remove
#     '21334238': 0.025,  # TODO remove
#     '21293326': 0,
#     '21293325': 0
# }

CAM_WEIGHTS_LH_CELLO = {
    '21334181': 0.65,
    '21334237': 0.,
    '21334190': 0.3,
    '21334211': 0.,
    '21334180': 0.05,
    '21334209': 0.,
    '21334221': 0.,
    '21334219': 0.,
    '21334208': 0.,
    '21334186': 0.,
    '21334184': 0.,  # TODO remove
    '21334238': 0.,  # TODO remove
    '21293326': 0.,
    '21293325': 0.
}

# CAM_WEIGHTS_RH_CELLO = {
#     '21334181': 0.30,
#     '21334237': 0.10,
#     '21334190': 0.10,
#     '21334211': 0.10,
#     '21334180': 0.10,
#     '21334209': 0.10,
#     '21334221': 0.05,
#     '21334219': 0.025,
#     '21334208': 0.025,
#     '21334186': 0.025,
#     '21334184': 0,
#     '21334238': 0.025,
#     '21293326': 0.025,
#     '21293325': 0.025
# }

CAM_WEIGHTS_RH_CELLO = {
    '21334181': 0.65,
    '21334237': 0.,
    '21334190': 0.3,
    '21334211': 0.,
    '21334180': 0.05,
    '21334209': 0.,
    '21334221': 0.,
    '21334219': 0.,
    '21334208': 0.,
    '21334186': 0.,
    '21334184': 0.,  # TODO remove
    '21334238': 0.,  # TODO remove
    '21293326': 0.,
    '21293325': 0.
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
    parser.add_argument('--summary_jsonfile',
                        default='../data/cello/cello01/cello01_summary.json',
                        type=str, 
                        required=True)
    parser.add_argument('--parent_dir', default='cello', type=str, required=True)
    parser.add_argument('--proj_dir', default='cello01', type=str, required=True)
    parser.add_argument('--start_frame', default='128', type=int, required=True)
    parser.add_argument('--end_frame', default='786', type=int, required=False)
    parser.add_argument('--instrument', default='cello', type=str, required=True)
    parser.add_argument('--cam_num', default='cam0', type=str, required=True)
    parser.add_argument('--end_frame', default='786', type=int, required=False)
    parser.add_argument('--mano_file_appendix', default='CUSTOMED', type=str, required=False)
    
    args = parser.parse_args()
    with open(args.summary_jsonfile,'r') as f:
        summary = json.load(f)
    f.close()
    
    parent_dir = args.parent_dir
    proj_dir = args.proj_dir
    start_frame = args.start_frame
    end_frame = args.end_frame
    instrument = args.instrument
    cam_num = args.cam_num
    mano_file_appendix = args.mano_file_appendix
    
    cam_param = summary['CameraParameter']
    cam_drop_frames = summary['FrameDropIDLog']
    
    start_frame = summary['StartFrame']
    end_frame = summary['EndFrame']

    dir_6d = f"./6d_result/{parent_dir}/{proj_dir}"

    lh_bone_length = get_lh_bone_length(proj_dir)

    with open(f'../audio/cp_result/{parent_dir}/{proj_dir}/kp_3d_all_dw_cp.json', 'r') as f:
        data_dict = json.load(f)
    kp_3d_all = np.array(data_dict['kp_3d_all_dw_cp'])
    
    # bone_lengths = get_bone_length_dw(kp_3d_all, 1)
    
    rh_bone_length = get_bone_length_dw(kp_3d_all, 1)[1]
    bone_lengths = np.array([lh_bone_length, rh_bone_length])
    
    integrated_hand_rot = []
    for frame in range(len(kp_3d_all)):
        lh_wrist = kp_3d_all[frame][LEFT_WRIST_INDEX]
        rh_wrist = kp_3d_all[frame][RIGHT_WRIST_INDEX]
        lh_joints, rh_joints, lh_rot, rh_rot = get_hands_joints(dir_6d, frame + start_frame, 
                                                                bone_lengths, cam_param, 
                                                                cam_num, instrument, 
                                                                parent_dir, proj_dir,
                                                                cam_drop_frames, mano_file_appendix)
        
        hand_rot = np.concatenate([lh_rot, rh_rot], axis=0)
        integrated_hand_rot.append(hand_rot.tolist())

        lh_joints_pano = lh_joints + lh_wrist
        rh_joints_pano = rh_joints + rh_wrist

        # kp_3d_all[frame][LEFT_WRIST_INDEX:LEFT_WRIST_INDEX+42] = np.vstack((lh_joints_pano, rh_joints_pano))
        kp_3d_all[frame][LEFT_WRIST_INDEX:RIGHT_WRIST_INDEX] = lh_joints_pano
        kp_3d_all[frame][RIGHT_WRIST_INDEX:RIGHT_WRIST_INDEX+21] = rh_joints_pano
        
        print(f'Hand pose integrated for frame {frame + start_frame}. -> [{proj_dir}]')
        

    if not os.path.exists(f'./fk_result/{parent_dir}/{proj_dir}'):
        os.makedirs(f'./fk_result/{parent_dir}/{proj_dir}', exist_ok=True)

    data_dict = {'kp_3d_all_pe_fk': kp_3d_all.tolist()}
    with open(f'./fk_result/{parent_dir}/{proj_dir}/kp_3d_all_pe_fk.json', 'w') as f:
        json.dump(data_dict, f)

    data_dict = {'integrated_hand_rot': integrated_hand_rot}
    with open(f'./fk_result/{parent_dir}/{proj_dir}/integrated_hand_rot.json', 'w') as f:
        json.dump(data_dict, f)
