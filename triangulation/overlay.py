import os.path
import cv2
from icecream import ic
import json
import numpy as np
import matplotlib.pyplot as plt
import imageio
from contextlib import contextmanager
from triangulation import make_projection_matrix
from triangulation import HUMAN_LINKS, CELLO_LINKS, BOW_LINKS

STRING_LINKS = [[142, 143],
                [144, 145],
                [146, 147],
                [148, 149]]


@contextmanager
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


def visualize_demo(data):
    framenum = data.shape[0]

    if not os.path.exists(f'../reproj_demo/'):
        os.makedirs(f'../reproj_demo/')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'../reproj_demo/output.avi', fourcc, fps=30, frameSize=[2300, 2656])

    for f in range(framenum):
        ic(f)
        kp_2d = data[f]
        fig = plt.figure(figsize=[10, 10])
        axes = fig.add_subplot()

        img = imageio.v2.imread(f"../data/cello_1113/cello_1113_scale/frames/21334181/camera_21334181_{f + 128}.jpg")
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
        cv2.imwrite(f"../reproj_demo/sample{f}.jpg", img_with_plot)
        out.write(img_with_plot)
        plt.close()


if __name__ == "__main__":
    # with open('../kp_3d_result/cello_1113_scale/kp_3d_smooth.json', 'r') as f:
    #     data_dict = json.load(f)
    # kp_3d_all = np.array(data_dict['kp_3d_smooth'])

    with open('../audio/kp_3d_all_with_cp.json', 'r') as f:
        data_dict = json.load(f)
    kp_3d_all = np.array(data_dict['kp_3d_all_with_cp'])

    framenum = kp_3d_all.shape[0]
    kpt_num = kp_3d_all.shape[1]

    cam_file = "jsons/cello_1113_scale_camera.json"
    cam_param = json.load(open(cam_file))

    # find reprojection of the specific camera
    repro_2d = np.empty([framenum, kpt_num, 2])
    repro_2d.fill(np.nan)
    proj_mat_cam_x = make_projection_matrix(cam_param, cams=['cam0'])
    for ff in range(framenum):
        for kpt in range(kpt_num):
            ones = np.ones((1))
            kp4d = np.concatenate([kp_3d_all[ff][kpt], ones], axis=0)
            kp4d = kp4d.reshape(-1)
            # reprojection: p = mP
            kp2d = np.matmul(proj_mat_cam_x, kp4d)
            kp2d = kp2d.reshape((3,))
            kp2d = kp2d / kp2d[2:3]
            repro_2d[ff, kpt, :] = kp2d[:2]

    visualize_demo(repro_2d)
