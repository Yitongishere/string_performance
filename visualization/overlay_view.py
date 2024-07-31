import os.path
import cv2
from icecream import ic
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
import sys
import imageio
sys.path.append('..')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from triangulation.triangulation_pipeline import make_projection_matrix, BOW_LINKS, HUMAN_WITHOUT_LH_THUMB_LINKS, \
    LH_THUMB_LINKS
from triangulation.triangulation_pipeline import HUMAN_LINKS, CELLO_LINKS, STRING_LINKS
from tools.load_summary import get_folder, get_inform, get_folder_extra
from tools.rotate import frame_rotate
from multiprocessing import Pool
from tqdm import tqdm


REAL_CELLO_NUT_L_BRIDGE_L = 695
REAL_VIOLIN_NUT_L_BRIDGE_L = 328


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


def visualize_overlay(proj_path, overlay_cam, data, cam_num, video, start_frame = 1,parent_dir = None,saveframes = True):
    if parent_dir is None:
        parent_dir = proj_path[:-2]
    framenum = data.shape[0]
    
    if not os.path.exists(f'./reproj_result/{parent_dir}/{proj_path}/{overlay_cam}'):
        os.makedirs(f'./reproj_result/{parent_dir}/{proj_path}/{overlay_cam}', exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'./reproj_result/{parent_dir}/{proj_path}/{proj_path}_{overlay_cam}_overlay.avi', fourcc, fps=30, frameSize=[2300, 2656])
    
    for f in tqdm(range(framenum), desc=f'(Overlay -> "{proj_path}")'):
        kp_2d = data[f]
        
        #img = imageio.v2.imread(f"{img_path}_{f + start_frame}.jpg")
        #img_with_plot = img.copy()
        img = np.asarray(video.get_data(f + start_frame - 1), dtype=np.uint8)
        img = frame_rotate(cam_num, img)
        img_with_plot = img.copy()

        # TODO edit the violin indices
        with plot_over(img_with_plot) as axes:
            # human body
            axes.scatter(kp_2d[0:133, 0],
                         kp_2d[0:133, 1], c='#1f77b4', s=50, zorder=5)
            # cello body
            axes.scatter(kp_2d[133:140, 0],
                         kp_2d[133:140, 1], c='saddlebrown', s=50, zorder=13)
            axes.scatter(kp_2d[140:142, 0],
                         kp_2d[140:142, 1], c='cyan', s=50, zorder=13)
            axes.scatter(kp_2d[142:150, 0],
                         kp_2d[142:150, 1], c='w', s=50, zorder=13)
            # axes.scatter(kp_2d[142, 0],
            #              kp_2d[142, 1], c='r', s=50,
            #              zorder=100)
            if True not in np.isnan(kp_2d[150]):
                axes.scatter(kp_2d[150, 0],
                             kp_2d[150, 1], c='r', s=50,
                             zorder=100)  # zorder must be the biggest so that it would not be occluded
                # axes.scatter(kp_2d[147:152, 0],
                #              kp_2d[147:152, 1], c='orange', s=50,
                #              zorder=99)
            # else:
            #     print(f'Frame {f} contact point not exist.')

            for human in HUMAN_WITHOUT_LH_THUMB_LINKS:
                plt.plot([kp_2d[human[0]][0], kp_2d[human[1]][0]], [kp_2d[human[0]][1], kp_2d[human[1]][1]], c='b',
                         linewidth=3, zorder=11)
            for human in LH_THUMB_LINKS:
                plt.plot([kp_2d[human[0]][0], kp_2d[human[1]][0]], [kp_2d[human[0]][1], kp_2d[human[1]][1]],
                         c='b',
                         linewidth=3, zorder=11)
            for cello in CELLO_LINKS:
                plt.plot([kp_2d[cello[0]][0], kp_2d[cello[1]][0]], [kp_2d[cello[0]][1], kp_2d[cello[1]][1]],
                         c='saddlebrown', zorder=10)
            for bow in BOW_LINKS:
                plt.plot([kp_2d[bow[0]][0], kp_2d[bow[1]][0]], [kp_2d[bow[0]][1], kp_2d[bow[1]][1]], c='cyan', zorder=10)

            for string in STRING_LINKS:
                plt.plot([kp_2d[string[0]][0], kp_2d[string[1]][0]], [kp_2d[string[0]][1], kp_2d[string[1]][1]], c='w', zorder=10)

        img_with_plot = img_with_plot[:, :, ::-1]
        # vis = cv2.resize(img_with_plot, (1150, 1328))
        # cv2.imshow('test', vis)
        # cv2.waitKey(0)
        
        #Save
        #cv2.imwrite(f"./reproj_result/{parent_dir}/{proj_path}/{overlay_cam}/{f + start_frame - 1}.jpg", img_with_plot)
        out.write(img_with_plot)
        plt.close()
    
    if not saveframes:
        os.removedirs(f'./reproj_result/{parent_dir}/{proj_path}/{overlay_cam}')


def overlay_process(proj_dir):
    root_path = os.path.abspath(f'../data/{parent_dir}')
    summary, summary_jsonfile_path = get_inform(proj_dir,root_path)
    
    overlay_cam = '21334181'
    cam_num = 'cam0'
    
    start_frame = summary['StartFrame'] # cello01 -> 128
    
    video_path = os.path.join(root_path,proj_dir,f'{proj_dir}_{overlay_cam}.avi')
    video = imageio.get_reader(os.path.abspath(video_path), 'ffmpeg')
    cam_param = summary['CameraParameter']

    with open(f'../pose_estimation/ik_result/{parent_dir}/{proj_dir}/kp_3d_ik_smooth.json', 'r') as f:
        data_dict = json.load(f)
    kp_3d_all = np.array(data_dict['kp_3d_ik_smooth'])

    with open(f'../audio/cp_result/{parent_dir}/{proj_dir}/kp_3d_all_dw_cp.json', 'r') as f:
        data_dict = json.load(f)
    kp_3d_dw = np.array(data_dict['kp_3d_all_dw_cp'])

    # first frame is labeled manually
    manual_label = kp_3d_dw[0]

    label_nut_l = manual_label[134]
    label_bridge_l = manual_label[136]

    label_nut_l_bridge_l = math.dist(label_nut_l, label_bridge_l)

    if instrument == 'cello':
        real_nul_l_bridge_l = REAL_CELLO_NUT_L_BRIDGE_L
    elif instrument == 'violin':
        real_nul_l_bridge_l = REAL_VIOLIN_NUT_L_BRIDGE_L
    else:
        raise Exception('Instrument type is not supported, please modify it into "cello" or "violin"!')

    ratio = real_nul_l_bridge_l / label_nut_l_bridge_l

    kp_3d_all /= ratio

    framenum = kp_3d_all.shape[0]
    kpt_num = kp_3d_all.shape[1]
    
    # find reprojection of the specific camera
    repro_2d = np.empty([framenum, kpt_num, 2])
    repro_2d.fill(np.nan)
    proj_mat_cam_x = make_projection_matrix(cam_param, cams=[cam_num])  # change here for various perspectives
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

    visualize_overlay(proj_dir, overlay_cam, repro_2d, overlay_cam, video, start_frame, parent_dir = parent_dir, saveframes = False)
    print(f'{proj_dir} -> successful')
    print('-'*60)


if __name__ == "__main__":
    parent_dir = 'cello'
    instrument = 'cello'
    proj_range = range(1,86)
    proj_dirs = [parent_dir+'0'*(2-len(str(i)))+str(i) for i in proj_range]
    print(proj_dirs)
    
    with Pool(processes=os.cpu_count()) as pool: 
        pool.map(overlay_process, proj_dirs)
