import numpy as np
from scipy import signal


def Kalman_filter(data, jointnum, KalmanParamQ=0.001, KalmanParamR=0.0015):
    framenum = data.shape[0]
    kalman_result = np.zeros_like(data)
    for i in range(framenum):
        smooth_kps = np.zeros((jointnum, 3), dtype=np.float32)

        K = np.zeros((jointnum, 3), dtype=np.float32)
        P = np.zeros((jointnum, 3), dtype=np.float32)
        X = np.zeros((jointnum, 3), dtype=np.float32)
        for j in range(jointnum):
            K[j] = (P[j] + KalmanParamQ) / (P[j] + KalmanParamQ + KalmanParamR)
            P[j] = KalmanParamR * (P[j] + KalmanParamQ) / (P[j] + KalmanParamQ + KalmanParamR)
        for j in range(jointnum):
            smooth_kps[j] = X[j] + (data[i][j] - X[j]) * K[j]
            X[j] = smooth_kps[j]
        kalman_result[i] = smooth_kps  # record kalman result
    return kalman_result


def Lowpass_Filter(data, jointnum, LowPassParam=0.1):
    lowpass_result = np.zeros_like(data)
    # take 6 previous frames for the smooth
    PrevPos3D = np.zeros((6, jointnum, 3), dtype=np.float32)
    framenum = data.shape[0]
    for i in range(framenum):
        PrevPos3D[0] = data[i]
        for j in range(1, 6):
            PrevPos3D[j] = PrevPos3D[j] * LowPassParam + PrevPos3D[j - 1] * (1.0 - LowPassParam)
        lowpass_result[i] = PrevPos3D[5]  # record lowpass result
    # lowpass_result[0] = lowpass_result[1]  # first frame lacks of prior info (remove it otherwise deviation occurs)
    return lowpass_result


# def Savgol_Filter(data, jointnum, WindowLength=11, PolyOrder=5):
#     savgol_result = np.zeros_like(data)
#     for joint in range(jointnum):
#         data_joint_x = data[:, joint, 0]
#         data_joint_y = data[:, joint, 1]
#         data_joint_z = data[:, joint, 2]
#         savgol_result[:, joint, 0] = signal.savgol_filter(data_joint_x, WindowLength, PolyOrder)
#         savgol_result[:, joint, 1] = signal.savgol_filter(data_joint_y, WindowLength, PolyOrder)
#         savgol_result[:, joint, 2] = signal.savgol_filter(data_joint_z, WindowLength, PolyOrder)
#     # point 12 is most occluded
#     savgol_result[:, 11, 0] = signal.savgol_filter(data[:, 11, 0], 27, 3)
#     savgol_result[:, 11, 1] = signal.savgol_filter(data[:, 11, 1], 27, 3)
#     savgol_result[:, 11, 2] = signal.savgol_filter(data[:, 11, 2], 27, 3)
#     return savgol_result

def Savgol_Filter(data, jointnum, WindowLength=[11, 22, 44], PolyOrder=[6, 4, 2]):
    """Enhancing smoothing process with longer WindowLength or lower PolyOrder"""

    all_kps = [i for i in range(jointnum)]
    fine_kps = [i for i in range(91, 112)]  # left hand
    fine_kps.append(9)  # left wrist
    ordinary_kps = list(set(all_kps).difference(set(fine_kps)))
    occluded_kps = [11, 12]  # occluded hip joints
    savgol_result = np.zeros_like(data)

    # fine key points with shorter WindowLength and higher PolyOrder
    for f_joint in fine_kps:
        savgol_result[:, f_joint, 0] = signal.savgol_filter(data[:, f_joint, 0], WindowLength[0], PolyOrder[0])
        savgol_result[:, f_joint, 1] = signal.savgol_filter(data[:, f_joint, 1], WindowLength[0], PolyOrder[0])
        savgol_result[:, f_joint, 2] = signal.savgol_filter(data[:, f_joint, 2], WindowLength[0], PolyOrder[0])

    for joint in ordinary_kps:
        savgol_result[:, joint, 0] = signal.savgol_filter(data[:, joint, 0], WindowLength[1], PolyOrder[1])
        savgol_result[:, joint, 1] = signal.savgol_filter(data[:, joint, 1], WindowLength[1], PolyOrder[1])
        savgol_result[:, joint, 2] = signal.savgol_filter(data[:, joint, 2], WindowLength[1], PolyOrder[1])

    # enhance the smoothing process towards occluded_kps
    for o_joint in occluded_kps:
        savgol_result[:, o_joint, 0] = signal.savgol_filter(data[:, o_joint, 0], WindowLength[2], PolyOrder[2])
        savgol_result[:, o_joint, 1] = signal.savgol_filter(data[:, o_joint, 1], WindowLength[2], PolyOrder[2])
        savgol_result[:, o_joint, 2] = signal.savgol_filter(data[:, o_joint, 2], WindowLength[2], PolyOrder[2])
    return savgol_result

