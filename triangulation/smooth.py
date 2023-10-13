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

def Savgol_Filter(data, jointnum, WindowLength=11, PolyOrder=5):
    savgol_result = np.zeros_like(data)
    for joint in range(jointnum):
        data_joint_x = data[:, joint, 0]
        data_joint_y = data[:, joint, 1]
        data_joint_z = data[:, joint, 2]
        savgol_result[:, joint, 0] = signal.savgol_filter(data_joint_x, WindowLength, PolyOrder)
        savgol_result[:, joint, 1] = signal.savgol_filter(data_joint_y, WindowLength, PolyOrder)
        savgol_result[:, joint, 2] = signal.savgol_filter(data_joint_z, WindowLength, PolyOrder)
    return savgol_result

if __name__ == '__main__':
    a = np.array([1,2,3,4,5,6,7,8]).reshape(2,2,2)
    b = np.zeros_like(a)
    b[:,0,:] = a[:,0,:]
    c = [1,2,3]
    print(c[1:3])
    print(b)
