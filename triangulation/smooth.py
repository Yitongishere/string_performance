import numpy as np


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


if __name__ == '__main__':
    a = np.array([1, 2, 3, 4, 5, 6]).reshape([1, 2, 3])
    print(np.squeeze(a[0]))
