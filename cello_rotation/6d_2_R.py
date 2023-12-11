import numpy as np
from pyquaternion import Quaternion
from icecream import ic


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


if __name__ == "__main__":
    filepath = "./1113_scale/cello_1113_21293325/1.txt"

    with open(filepath) as f:
        lines = f.readlines()

    lh = np.array([float(x) for x in lines[1].rstrip().split(' ')]).reshape(16, 6)
    rh = np.array([float(x) for x in lines[3].rstrip().split(' ')]).reshape(16, 6)

    R_lh = rotation_6d_to_R(lh)
    R_rh = rotation_6d_to_R(rh)
    # ic(R_lh)
    ic(R_lh.shape)

    for i in range(len(R_lh)):
        ic(is_orthogonal(R_lh[i]))
        ic(is_orthogonal(R_rh[i]))
