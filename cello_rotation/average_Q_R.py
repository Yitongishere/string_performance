import numpy as np
import os
from preprocess_6d import get6d_from_txt, rotation_6d_to_R, rotationR_to_Q
from scipy.spatial.transform import Rotation, Slerp
from pyquaternion import Quaternion
from icecream import ic

def quaternion_norm(q):
    return np.sqrt(np.sum(np.square(q)))

def get_frame_info(proj_dir, frame_num):

    cam_dirs = os.listdir(proj_dir)
    frame_info = {}
    for cam_dir in cam_dirs:
        cam_num = cam_dir.split('_')[-1]
        filepath = os.path.join(proj_dir, cam_dir, f'{frame_num}.txt')

        lh, rh = get6d_from_txt(filepath)

        R_lh = rotation_6d_to_R(lh)
        R_rh = rotation_6d_to_R(rh)

        Q_lh = rotationR_to_Q(R_lh)
        Q_rh = rotationR_to_Q(R_rh)

        frame_info[f'{cam_num}'] = {'R_lh': R_lh,
                                    'R_rh': R_rh,
                                    'Q_lh': Q_lh,
                                    'Q_rh': Q_rh}

    return frame_info

def weighted_average_quaternion(q1, q2, w):
    key_rots = Rotation.from_quat((q1, q2))

    # 创建 Slerp 对象
    slerp = Slerp([0, 1], key_rots)

    # 进行球面线性插值
    interpolated_quaternion = slerp(w)

    return interpolated_quaternion.as_quat()


if __name__ == "__main__":

    # set camera weights in Q averaging
    cam_weights_lh = {
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

    cam_weights_rh = {
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

    cams = list(cam_weights_lh.keys())

    # 设定目标帧
    frame_num = 101
    proj_dir = "./1113_scale"
    frame_info = get_frame_info(proj_dir, frame_num)

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
                q_current_lh = weighted_average_quaternion(q_current_lh, q_next_lh, alpha_lh)
            if not np.isnan(q_next_rh).any():
                q_current_rh = weighted_average_quaternion(q_current_rh, q_next_rh, alpha_rh)

            weights_lh += cam_weights_lh[cams[c]]
            weights_rh += cam_weights_rh[cams[c]]

        averaged_Qs_lh.append(q_current_lh)
        averaged_Qs_rh.append(q_current_rh)

    # 创建旋转对象，q 转 r
    rots_lh = Rotation.from_quat(averaged_Qs_lh)
    rots_rh = Rotation.from_quat(averaged_Qs_rh)
    R_matrix_lh = rots_lh.as_matrix()
    R_matrix_rh = rots_rh.as_matrix()


    # 验证两种 q - r 转换的一致性
    # 但是 pyquaternion 对四元数的表示顺序为：w, x, y, z
    # 而 scipy.spatial.transform 的顺序为：x, y, z, w
    for i in range(len(R_matrix_lh)):
        q_l = np.array(list(Quaternion(matrix=R_matrix_lh[i])))
        q_r = np.array(list(Quaternion(matrix=R_matrix_rh[i])))
        # (w, x, y, z)
        ic(q_l)
        # ic(q_r)
    # (x, y, z, w)
    ic(averaged_Qs_lh)
    # ic(averaged_Qs_rh)

    print(f"第{frame_num}帧，左手加权平均后的R_lh为：\n {R_matrix_lh}")
    print(f"第{frame_num}帧，有手加权平均后的R_rh为：\n {R_matrix_rh}")







