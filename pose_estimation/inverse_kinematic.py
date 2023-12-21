import contextlib
import cv2
import imageio
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from handpose_toolkit import get6d_from_txt, rotation_6d_to_R, get_joint_positions, get_mano_init, get_converted_R0
from integrate_handpose import get_bone_length_dw, MANO_PARENTS_INDICES, LEFT_WRIST_INDEX, MANO_TO_DW
import json
from icecream import ic
from triangulation.triangulation import make_projection_matrix, HUMAN_LINKS, CELLO_LINKS, BOW_LINKS, STRING_LINKS



@contextlib.contextmanager
def plot_over(img, extent=None, origin="upper", dpi=100):
    """用于基于原图画点"""
    h, w, d = img.shape
    assert d == 3
    if extent is None:
        xmin, xmax, ymin, ymax = -0.5, w + 0.5, -0.5, h + 0.5
    else:
        xmin, xmax, ymin, ymax = extent
    if origin == "upper":
        ymin, ymax = ymax, ymin
    elif origin != "lower":
        raise ValueError("origin must be 'upper' or 'lower'")
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = plt.Axes(fig, (0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.add_axes(ax)
    fig.set_facecolor((0, 0, 0, 0))
    yield ax
    fig.canvas.draw()
    plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    rgb = plot[..., :3]
    alpha = plot[..., 3, None]
    img[...] = ((255 - alpha) * img.astype(np.uint16) + alpha * rgb.astype(np.uint16)) // 255


def visualize_overlay(kp_2d):

    # if not os.path.exists(f'../reproj_result/'):
    #     os.makedirs(f'../reproj_result/')
    #
    # if not os.path.exists(f'../reproj_result/{proj_path}/'):
    #     os.makedirs(f'../reproj_result/{proj_path}/')
    #
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(f'../reproj_result/{proj_path}/output.avi', fourcc, fps=30, frameSize=[2300, 2656])
    fig = plt.figure(figsize=[10, 10])
    axes = fig.add_subplot()

    img = imageio.v2.imread(f"../data/cello_1113/cello_1113_scale/frames/21334181/camera_21334181_{130}.jpg")
    img_with_plot = img.copy()
    with plot_over(img_with_plot) as axes:
        axes.scatter(kp_2d[0:133, 0],
                     kp_2d[0:133, 1], s=50)
        axes.scatter(kp_2d[133:140, 0],
                     kp_2d[133:140, 1], c='saddlebrown', s=50)
        axes.scatter(kp_2d[140:142, 0],
                     kp_2d[140:142, 1], c='goldenrod', s=50)
        axes.scatter(kp_2d[142:150, 0],
                     kp_2d[142:150, 1], c='w', s=50)
        if True not in np.isnan(kp_2d[150]):
            axes.scatter(kp_2d[150, 0],
                         kp_2d[150, 1], c='r', s=50,
                         zorder=100)  # zorder must be the biggest so that it would not be occluded
            axes.scatter(kp_2d[151:155, 0],
                         kp_2d[151:155, 1], c='orange', s=50,
                         zorder=99)
        else:
            print(f'Frame {f} contact point not exist.')

        for human in HUMAN_LINKS:
            plt.plot([kp_2d[human[0]][0], kp_2d[human[1]][0]], [kp_2d[human[0]][1], kp_2d[human[1]][1]], c='blue')
        for cello in CELLO_LINKS:
            plt.plot([kp_2d[cello[0]][0], kp_2d[cello[1]][0]], [kp_2d[cello[0]][1], kp_2d[cello[1]][1]],
                     c='saddlebrown')
        for bow in BOW_LINKS:
            plt.plot([kp_2d[bow[0]][0], kp_2d[bow[1]][0]], [kp_2d[bow[0]][1], kp_2d[bow[1]][1]], c='goldenrod')

        for string in STRING_LINKS:
            plt.plot([kp_2d[string[0]][0], kp_2d[string[1]][0]], [kp_2d[string[0]][1], kp_2d[string[1]][1]], c='w')

    img_with_plot = img_with_plot[:, :, ::-1]
    cv2.imwrite(f"./ik_result/ik_test.jpg", img_with_plot)
    img_with_plot = cv2.resize(img_with_plot, (1150, 1328))
    cv2.imshow('img', img_with_plot)
    cv2.waitKey(0)
    plt.close()


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


# TODO: Edit bone length passing way
def construct_jacobian(init_pos, init_rot_vec, contact_finger, bone_length):
    """
    Input:
    initial position (rest pose): (21 * 3)
    initial rotation vector: (16 * 3)
    contact finger index: (1,) 0-3 represents index, middle, ring and pinky
    Return:
    jocobian matrix: (6 * 3)
    """
    delta_theta = 1e-5
    Jacob_matrix = np.zeros([3, 3])
    for i in range(3):
        init_rot_mat = get_rot_mat(init_rot_vec)
        origin_pos = get_joint_positions(init_pos, init_rot_mat, bone_length, MANO_PARENTS_INDICES)
        origin_pos = mano_to_dw(origin_pos, lh_wrist)
        origin_tip_pos = origin_pos[MANO_TIP[contact_finger]]
        new_rot_vec = init_rot_vec.copy()
        new_rot_vec[0][i] += delta_theta
        new_rot_mat = get_rot_mat(new_rot_vec)
        new_pos = get_joint_positions(init_pos, new_rot_mat, bone_length, MANO_PARENTS_INDICES)
        new_pos = mano_to_dw(new_pos, lh_wrist)
        new_tip_pos = new_pos[MANO_TIP[contact_finger]]
        delta_J = new_tip_pos - origin_tip_pos
        partial_derivative = delta_J / delta_theta
        Jacob_matrix[:, i] = partial_derivative
    return Jacob_matrix


def mano_to_dw(mano_pose, wrist):
    dw_pose = mano_pose.copy()
    for dict_id, mano_id in enumerate(MANO_TO_DW):
        dw_pose[dict_id, :] = mano_pose[mano_id, :]
        dw_pose[dict_id, :] = mano_pose[mano_id, :]
    dw_pose += wrist
    return dw_pose


DW_TIP = [8, 12, 16, 20]
MANO_TIP = [16, 17, 19, 18]
file_path = '6d_result/cello_1113_scale/cello_1113_21334181/132.txt'

if __name__ == '__main__':
    left_hand, right_hand = get6d_from_txt(file_path)
    lh_rot = rotation_6d_to_R(left_hand)
    R0 = lh_rot[0]
    converted_R0 = get_converted_R0('cam0', R0)
    lh_rot[0] = converted_R0
    lh_rot_vec = get_rot_vec(lh_rot)

    proj_dir = "cello_1113_scale"
    dir_6d = f"./6d_result/{proj_dir}"
    frame_id = 4

    with open(f'../audio/{proj_dir}/kp_3d_all_dw_cp_smooth.json', 'r') as f:
        data_dict = json.load(f)
    kp_3d = np.array(data_dict['kp_3d_all_dw_cp_smooth'])

    lh_wrist = kp_3d[frame_id][LEFT_WRIST_INDEX]

    extracted_frame = kp_3d[frame_id]
    start_pos = kp_3d[frame_id][91:112]
    # ic(start_pos)

    init_pos = get_mano_init('left')

    cp = kp_3d[frame_id][150]
    # ic(cp)

    bls = get_bone_length_dw(kp_3d, 1)
    bl = bls[0]  # IK only involves left hand

    # TODO: get the contact finger index

    lr = 1e-7
    iter_times = 1000
    iter_pos_dw = start_pos.copy()
    iter_rot_vec = lh_rot_vec.copy()  # iteration init for rotation vector
    prev_r = [np.inf, np.inf, np.inf]
    for i in range(iter_times):
        tip = iter_pos_dw[DW_TIP[0]]
        r = abs(cp - tip)
        ic(r)
        # if i % 5 == 0 and np.linalg.norm(prev_r) < np.linalg.norm(r):
        #     print(i)
        #     break
        prev_r = r
        jm = construct_jacobian(init_pos, iter_rot_vec, 0, bl)
        # ic(jm)
        # ic(np.linalg.inv(jm))
        delta_rot_vec = np.matmul(np.linalg.inv(jm), r)
        ic(lr * delta_rot_vec)
        iter_rot_vec[0] = iter_rot_vec[0] - lr * delta_rot_vec  # Update R0
        iter_rot_mat = get_rot_mat(iter_rot_vec)
        iter_pos_mano = get_joint_positions(init_pos, iter_rot_mat, bone_lengths=bl, parent_indices=MANO_PARENTS_INDICES)
        iter_pos_dw = mano_to_dw(iter_pos_mano, lh_wrist)
        # iter_pos_dw = iter_pos_mano.copy()
        # for dict_id, mano_id in enumerate(MANO_TO_DW):
        #     iter_pos_dw[dict_id, :] = iter_pos_mano[mano_id, :]
        #     iter_pos_dw[dict_id, :] = iter_pos_mano[mano_id, :]
        # iter_pos_dw += lh_wrist

    extracted_frame[91:112] = iter_pos_dw
    ic(extracted_frame.shape)

    cam_file = "../triangulation/jsons/cello_1113_scale_camera.json"
    cam_param = json.load(open(cam_file))

    repro_2d = np.empty([155, 2])
    repro_2d.fill(np.nan)
    proj_mat_cam_x = make_projection_matrix(cam_param, cams=['cam0'])  # change here for various perspectives
    for kpt in range(155):
        ones = np.ones((1))
        kp4d = np.concatenate([extracted_frame[kpt], ones], axis=0)
        kp4d = kp4d.reshape(-1)
        # reprojection: p = mP
        kp2d = np.matmul(proj_mat_cam_x, kp4d)
        kp2d = kp2d.reshape((3,))
        kp2d = kp2d / kp2d[2:3]
        repro_2d[kpt, :] = kp2d[:2]

    # print(repro_2d)

    visualize_overlay(repro_2d)



