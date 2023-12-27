import contextlib
import os.path
import cv2
import imageio
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
from handpose_toolkit import get6d_from_txt, rotation_6d_to_R, get_joint_positions, get_mano_init, get_converted_R0, \
    cal_dist, weighted_average_quaternion
from integrate_handpose import get_bone_length_dw, MANO_PARENTS_INDICES, LEFT_WRIST_INDEX, MANO_TO_DW
import json
from icecream import ic

from triangulation.smooth import Savgol_Filter
from triangulation.triangulation import make_projection_matrix, HUMAN_LINKS, CELLO_LINKS, BOW_LINKS, STRING_LINKS, \
    visualize_3d
from scipy.optimize import minimize


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


def visualize_overlay(proj_dir, data):
    framenum = data.shape[0]
    # if not os.path.exists(f'../reproj_result/'):
    #     os.makedirs(f'../reproj_result/')
    #
    # if not os.path.exists(f'../reproj_result/{proj_path}/'):
    #     os.makedirs(f'../reproj_result/{proj_path}/')
    #
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'./ik_result/{proj_dir}/output_ik_smooth.avi', fourcc, fps=30, frameSize=[2300, 2656])

    for f in range(framenum):
        print(f'Frame {f} ik image generating...')
        kp_2d = data[f]
        fig = plt.figure(figsize=[10, 10])
        axes = fig.add_subplot()
        img = imageio.v2.imread(f"../data/cello_1113/cello_1113_scale/frames/21334181/camera_21334181_{f + 128}.jpg")
        img_with_plot = img.copy()
        with plot_over(img_with_plot) as axes:
            # axes.scatter(kp_2d[0:133, 0],
            #              kp_2d[0:133, 1], s=50)
            axes.scatter(kp_2d[0:92, 0],
                         kp_2d[0:92, 1], c='#1f77b4', s=50, zorder=2)
            axes.scatter(kp_2d[92:96, 0],
                         kp_2d[92:96, 1], c='grey', s=50, zorder=1)
            axes.scatter(kp_2d[96:133, 0],
                         kp_2d[96:133, 1], c='#1f77b4', s=50, zorder=2)
            axes.scatter(kp_2d[133:140, 0],
                         kp_2d[133:140, 1], c='saddlebrown', s=50)
            axes.scatter(kp_2d[140:142, 0],
                         kp_2d[140:142, 1], c='goldenrod', s=50)
            axes.scatter(kp_2d[142:150, 0],
                         kp_2d[142:150, 1], c='w', s=50)
            if True not in np.isnan(kp_2d[150]):
                axes.scatter(kp_2d[150, 0],
                             kp_2d[150, 1], c='r', s=25,
                             zorder=100)  # zorder must be the biggest so that it would not be occluded
                axes.scatter(kp_2d[151:155, 0],
                             kp_2d[151:155, 1], c='orange', s=0,
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
        cv2.imwrite(f"./ik_result/{proj_dir}/ik_{f + 128}.jpg", img_with_plot)
        # img_with_plot = cv2.resize(img_with_plot, (1150, 1328))
        # cv2.imshow('img', img_with_plot)
        # cv2.waitKey(0)
        out.write(img_with_plot)
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


def obj_func(rot_vec_wrist, rot_vec_except_wrist, wrist_pos, cp_pos, finger_id, bone_length):
    # ic(rot_vec_wrist.shape)
    # ic(rot_vec_except_wrist.shape)
    rot_vec = np.vstack((rot_vec_wrist, rot_vec_except_wrist))
    rot_mat = get_rot_mat(rot_vec)
    pos_mano = get_joint_positions(init_pos, rot_mat, bone_length, MANO_PARENTS_INDICES)
    pos_dw = mano_to_dw(pos_mano, wrist_pos)
    tip_pos = pos_dw[DW_TIP[finger_id]]
    dist_squared = np.sum((tip_pos - cp_pos) ** 2)
    return dist_squared


def find_finger(frame_num, kp_3d):
    ori_tip_position = kp_3d[frame_num][-1]
    distances = []
    for i in range(len(DW_TIPS_INDICES)):
        tip_i_position = kp_3d[frame_num][DW_TIPS_INDICES[i]]
        dis_i = cal_dist(ori_tip_position, tip_i_position)
        distances.append(dis_i)
    used_finger_id = distances.index(min(distances))
    return used_finger_id


def linear_interpolation(a_frame, a_displacement, b_frame, b_displacement, c_frame):
    # 计算 a 到 b 的总帧数
    total_frames = b_frame - a_frame
    # 计算待求解的帧 c 在总帧数中的位置
    c_position = c_frame - a_frame
    # 计算 c 的位移（线性插值）
    c_displacement = a_displacement + (b_displacement - a_displacement) * np.log((c_position / total_frames) + 1)
    return c_displacement


DW_TIP = [8, 12, 16, 20]
MANO_TIP = [16, 17, 19, 18]
DW_TIPS_INDICES = [99, 103, 107, 111]
if __name__ == '__main__':
    proj_dir = "cello_1113_scale"
    dir_6d = f"./6d_result/{proj_dir}"

    with open(f'../audio/{proj_dir}/kp_3d_all_dw_cp.json', 'r') as f:
        data_dict = json.load(f)
    kp_3d_dw = np.array(data_dict['kp_3d_all_dw_cp'])

    with open(f'../pose_estimation/{proj_dir}/kp_3d_all_pe.json', 'r') as f:
        data_dict = json.load(f)
    kp_3d_pe = np.array(data_dict['kp_3d_all_pe'])

    kp_3d_ik = kp_3d_dw.copy()

    if not os.path.exists('./ik_result'):
        os.mkdir('./ik_result')
    if not os.path.exists(f'./ik_result/{proj_dir}'):
        os.mkdir(f'./ik_result/{proj_dir}')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(f'./ik_result/{proj_dir}/output_ik.avi', fourcc, fps=30, frameSize=[2300, 2656])

    dir6d_path = '6d_result/cello_1113_scale/cello_1113_21334181/'
    find_first_cp = 0
    for e, frame_id in enumerate(range(kp_3d_pe.shape[0])):
        # for e, frame_id in enumerate([37, 38, 39, 40, 41, 42, 43, 44, 45]):
        ic(frame_id)
        file_path = os.path.join(dir6d_path, f'{128 + frame_id}.txt')
        left_hand, right_hand = get6d_from_txt(file_path)
        lh_rot = rotation_6d_to_R(left_hand)
        R0 = lh_rot[0]
        converted_R0 = get_converted_R0('cam0', R0)
        lh_rot[0] = converted_R0
        lh_rot_vec = get_rot_vec(lh_rot)
        # ic(lh_rot_vec.shape)
        lh_rot_vec_wrist = lh_rot_vec[0]
        lh_rot_vec_except_wrist = lh_rot_vec[1:]
        lh_wrist = kp_3d_pe[frame_id][LEFT_WRIST_INDEX]

        extracted_frame = kp_3d_pe[frame_id]
        start_pos = kp_3d_pe[frame_id][91:112]
        init_pos = get_mano_init('left')

        cp = kp_3d_dw[frame_id][150]
        bls = get_bone_length_dw(kp_3d_pe, 1)
        bl = bls[0]  # IK only involves left hand
        finger = find_finger(frame_id, kp_3d_dw)

        ik_flag = 0
        frame_next = frame_id
        # 对于有cp的帧
        if not np.isnan(cp).any():
            if e == 0: find_first_cp = 1
            print(f"frame {frame_id} have cp")
            ik_flag = 1
            result = minimize(obj_func, lh_rot_vec_wrist,
                              args=(lh_rot_vec_except_wrist, lh_wrist, cp, finger, bl),
                              method='L-BFGS-B')
            optimized_rotvec_wrist = result.x

            init_R0 = Rotation.from_rotvec(lh_rot_vec_wrist).as_matrix()
            final_R0 = Rotation.from_rotvec(optimized_rotvec_wrist).as_matrix()
            diff_R0 = np.dot(final_R0, init_R0.T)

            frame_previous = frame_id
            previous_diff_R0 = diff_R0
            previous_rotvec = Rotation.from_matrix(final_R0).as_rotvec()

        # 对于没有cp的帧操作
        cp_n = 0
        while not ik_flag:
            print(f"frame {frame_id} does not have cp")
            frame_next += 1
            # ic(frame_next)
            # 对于结尾帧
            if frame_next == kp_3d_pe.shape[0]:
                current_diff_R0 = previous_diff_R0
                # 当前帧的手腕旋转信息
                lh_rot_mat_wrist = Rotation.from_rotvec(lh_rot_vec_wrist).as_matrix()
                current_final_R0 = np.dot(current_diff_R0, lh_rot_mat_wrist)
                optimized_rotvec_wrist = Rotation.from_matrix(current_final_R0).as_rotvec()
                break
            cp_next = kp_3d_dw[frame_next][150]
            if not np.isnan(cp_next).any():
                # 找到了下一个有cp的帧，此时的frame_next
                ik_flag = 1
                file_path_next = os.path.join(dir6d_path, f'{128 + frame_next}.txt')
                left_hand_next, _ = get6d_from_txt(file_path_next)
                lh_rot_next = rotation_6d_to_R(left_hand_next)
                R0_next = lh_rot_next[0]
                converted_R0_next = get_converted_R0('cam0', R0_next)
                lh_rot_next[0] = converted_R0_next
                lh_rot_vec_next = get_rot_vec(lh_rot_next)
                lh_rot_vec_wrist_next = lh_rot_vec_next[0]
                lh_rot_vec_except_wrist_next = lh_rot_vec_next[1:]
                lh_wrist_next = kp_3d_pe[frame_next][LEFT_WRIST_INDEX]
                cp_next = kp_3d_dw[frame_next][150]
                finger_next = find_finger(frame_next, kp_3d_dw)

                result_next = minimize(obj_func, lh_rot_vec_wrist_next,
                                       args=(lh_rot_vec_except_wrist_next, lh_wrist_next, cp_next, finger_next, bl),
                                       method='L-BFGS-B')
                optimized_rotvec_wrist_next = result_next.x

                init_R0_next = Rotation.from_rotvec(lh_rot_vec_wrist_next).as_matrix()
                final_R0_next = Rotation.from_rotvec(optimized_rotvec_wrist_next).as_matrix()
                diff_R0_next = np.dot(final_R0_next, init_R0_next.T)
                next_diff_R0 = diff_R0_next
                next_rotvec = Rotation.from_matrix(final_R0_next).as_rotvec()

                # 提前记录frame_next的位移，为插值translation准备
                next_finger = find_finger(frame_next, kp_3d_dw)
                next_optimized_rotvec = np.vstack((next_rotvec, lh_rot_vec_except_wrist))
                next_optimized_rotmat = get_rot_mat(next_optimized_rotvec)
                next_optimized_pos_mano = get_joint_positions(init_pos, next_optimized_rotmat, bl, MANO_PARENTS_INDICES)
                next_optimized_pos_dw = mano_to_dw(next_optimized_pos_mano, lh_wrist)
                next_optimized_tip = next_optimized_pos_dw[DW_TIP[finger_next]]
                next_translation = cp_next - next_optimized_tip

                # 非视频开头，现在有了 previous_diff_R0 和 next_diff_R0，基于当前帧frame_id,转换为Q做插值
                if find_first_cp:
                    previous_diff_Q = Rotation.from_matrix(previous_diff_R0).as_quat()
                    next_diff_Q = Rotation.from_matrix(next_diff_R0).as_quat()
                    current_diff_Q = weighted_average_quaternion(previous_diff_Q, next_diff_Q, frame_previous,
                                                                 frame_next, frame_id)
                    print(
                        f"No cp in frame {frame_id}, then interpolated rotation between {frame_previous} and {frame_next}.")
                    current_diff_R0 = Rotation.from_quat(current_diff_Q).as_matrix()
                    # 当前无cp帧的手腕旋转信息
                    lh_rot_mat_wrist = Rotation.from_rotvec(lh_rot_vec_wrist).as_matrix()
                    current_final_R0 = np.dot(current_diff_R0, lh_rot_mat_wrist)
                    optimized_rotvec_wrist = Rotation.from_matrix(current_final_R0).as_rotvec()
                    print(f"front base of interpolation frame {frame_previous} is {previous_rotvec}")
                    print(f"interpolated rotvec for frame {frame_id} is {optimized_rotvec_wrist}")
                    print(f"end base of interpolation frame {frame_next} is {next_rotvec}")

                # 对于视频开头第一个cp帧的之前帧
                frames_before_cp = []
                while (not find_first_cp and cp_n != 2):
                    ic(cp_n)
                    cp_n += 1
                    if cp_n == 1:
                        ik_flag = 0
                        diff_R0_cp1 = diff_R0_next
                        break
                    if cp_n == 2:
                        # ik_flag = 1
                        diff_R0_cp2 = diff_R0_next
                    # 计算第一帧cp和第二帧cp间，旋转变化差多少
                    diff_diff_cp1cp2_R = np.dot(diff_R0_cp2, diff_R0_cp1.T)
                    base_R_final = diff_R0_cp1
                    for i in range(frame_next - frame_id - 1):
                        current_diff_R0 = np.dot(base_R_final, diff_diff_cp1cp2_R.T)
                        base_R_final = current_diff_R0
                    # 当前无cp帧的手腕旋转信息
                    lh_rot_mat_wrist = Rotation.from_rotvec(lh_rot_vec_wrist).as_matrix()
                    current_final_R0 = np.dot(current_diff_R0, lh_rot_mat_wrist)
                    optimized_rotvec_wrist = Rotation.from_matrix(current_final_R0).as_rotvec()
                    frames_before_cp.append(frame_id)
                    if frame_next - frame_id == 2:
                        frames_before_first_cp = frame_id
                        print(f"第一个cp出现前有{frames_before_first_cp}帧")
                        find_first_cp = 1

        optimized_rotvec = np.vstack((optimized_rotvec_wrist, lh_rot_vec_except_wrist))
        optimized_rotmat = get_rot_mat(optimized_rotvec)
        optimized_pos_mano = get_joint_positions(init_pos, optimized_rotmat, bl, MANO_PARENTS_INDICES)
        optimized_pos_dw = mano_to_dw(optimized_pos_mano, lh_wrist)
        optimized_tip = optimized_pos_dw[DW_TIP[finger]]

        # 开头第一个cp前的帧
        if frame_id in frames_before_cp:
            cp = optimized_tip
        # 中间无cp帧
        if np.isnan(cp).any():
            pre_translation = translation
            translation = linear_interpolation(frame_previous, pre_translation, frame_next, next_translation, frame_id)
        # 有cp帧
        else:
            translation = cp - optimized_tip

        # 平移操作
        # optimized_pos_dw = optimized_pos_dw + translation  # translate to contact point position

        # # ic(optimized_pos_dw)
        extracted_frame[9] = optimized_pos_dw[0]  # 9 is also wrist
        extracted_frame[91:112] = optimized_pos_dw
        extracted_frame[112:133] = kp_3d_dw[frame_id][112:133]  # right hand should follow dw result

        extracted_frame = np.vstack((extracted_frame, kp_3d_dw[frame_id][142:]))

        kp_3d_ik[frame_id] = extracted_frame

        # cam_file = "../triangulation/jsons/cello_1113_scale_camera.json"
        # cam_param = json.load(open(cam_file))
        #
        # repro_2d = np.empty([155, 2])
        # repro_2d.fill(np.nan)
        # proj_mat_cam_x = make_projection_matrix(cam_param, cams=['cam0'])  # change here for various perspectives
        # for kpt in range(155):
        #     ones = np.ones((1))
        #     kp4d = np.concatenate([extracted_frame[kpt], ones], axis=0)
        #     kp4d = kp4d.reshape(-1)
        #     # reprojection: p = mP
        #     kp2d = np.matmul(proj_mat_cam_x, kp4d)
        #     kp2d = kp2d.reshape((3,))
        #     kp2d = kp2d / kp2d[2:3]
        #     repro_2d[kpt, :] = kp2d[:2]
        #
        # # print(repro_2d)
        #
        # img = visualize_overlay(proj_dir, repro_2d, frame_id)
        # out.write(img)

    # Bow Points are currently not available, otherwise index should be set to 142
    kp_3d_partial_ik = kp_3d_ik[:, :140, :]
    kp_3d_partial_ik_smooth = Savgol_Filter(kp_3d_partial_ik, 140)
    ic(kp_3d_partial_ik_smooth.shape)
    ik_cp = kp_3d_ik[:, 140:, :]
    kp_3d_ik_smooth = np.concatenate((kp_3d_partial_ik_smooth, ik_cp), axis=1)

    # 手指视角
    # visualize_3d(kp_3d_ik_smooth, proj_dir, 'pe_cp_ik_notrans', 'finger')

    data_dict = {'kp_3d_ik_smooth': kp_3d_ik_smooth.tolist()}
    if not os.path.exists(proj_dir):
        os.mkdir(proj_dir)
    with open(f'{proj_dir}/kp_3d_ik_smooth.json', 'w') as f:
        json.dump(data_dict, f)



    framenum = kp_3d_ik_smooth.shape[0]
    kpt_num = kp_3d_ik_smooth.shape[1]

    cam_file = "../triangulation/jsons/cello_1113_scale_camera.json"
    cam_param = json.load(open(cam_file))

    # find reprojection of the specific camera
    repro_2d = np.empty([framenum, kpt_num, 2])
    repro_2d.fill(np.nan)
    proj_mat_cam_x = make_projection_matrix(cam_param, cams=['cam0'])  # change here for various perspectives
    for ff in range(framenum):
        for kpt in range(kpt_num):
            ones = np.ones((1))
            kp4d = np.concatenate([kp_3d_ik_smooth[ff][kpt], ones], axis=0)
            kp4d = kp4d.reshape(-1)
            # reprojection: p = mP
            kp2d = np.matmul(proj_mat_cam_x, kp4d)
            kp2d = kp2d.reshape((3,))
            kp2d = kp2d / kp2d[2:3]
            repro_2d[ff, kpt, :] = kp2d[:2]

    visualize_overlay(proj_dir, repro_2d)




