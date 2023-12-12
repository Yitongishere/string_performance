import os
import numpy as np
from pyquaternion import Quaternion
from icecream import ic
import json

def get6d_from_txt(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    lh = np.array([float(x) for x in lines[1].rstrip().split(' ')]).reshape(16, 6)
    rh = np.array([float(x) for x in lines[3].rstrip().split(' ')]).reshape(16, 6)

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
    R = np.stack((b1, b2, b3), axis=-2)

    return R

def rotationR_to_Q(rotation_matrix):
    Q = np.full((16, 4), np.nan)
    if not np.isnan(rotation_matrix).any():
        for i in range(len(rotation_matrix)):
            Q[i] = np.array(list(Quaternion(matrix=rotation_matrix[i])))
    return Q


def numpy_arrays_to_lists(dic):
    for key, value in dic.items():
        if isinstance(value, np.ndarray):
            # 如果值是 NumPy 数组，转换为列表
            dic[key] = value.tolist()
        elif isinstance(value, dict):
            # 如果值是嵌套字典，递归调用此函数
            dic[key] = numpy_arrays_to_lists(value)
    return dic


if __name__ == "__main__":

    proj_dir = "./1113_scale"
    cam_dirs = os.listdir(proj_dir)
    frame = 5

    start_frame = 1
    end_frame = 83

    data_all = {}
    for frame in range(start_frame, end_frame+1):

        data_f = {}
        cams = []
        for cam_dir in cam_dirs:
            cam_num = cam_dir.split('_')[-1]
            cams.append(cam_num)

            filepath = os.path.join(proj_dir, cam_dir, f'{frame}.txt')
            # ic(filepath)

            lh, rh = get6d_from_txt(filepath)

            R_lh = rotation_6d_to_R(lh)
            R_rh = rotation_6d_to_R(rh)

            Q_lh = rotationR_to_Q(R_lh)
            Q_rh = rotationR_to_Q(R_rh)


            data_f[f'{cam_num}'] = {'R_lh': R_lh,
                                    'R_rh': R_rh,
                                    'Q_lh': Q_lh,
                                    'Q_rh': Q_rh}

        data_all[f'{frame}'] = data_f


    data_all = numpy_arrays_to_lists(data_all)
    with open('output.json', 'w') as json_file:
        json.dump(data_all, json_file, indent=4)

    # target_frame = '1'
    # for cam in cams:
    #     res = data_all[target_frame][cam]['Q_lh']
    #     print(f'The frame {target_frame}: Q_lh for cam {cam} is {res}')




