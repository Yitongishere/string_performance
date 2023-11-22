# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:37:43 2023

@author: shrim

This code will run TAPIR to track the keypoints of the cello that have labelled in *.json.
For more information, please visit "https://github.com/google-deepmind/tapnet".

The version of some wheels as below:
mediapy                         1.1.9
chex                            0.1.5
kubric                          0.1.1
jax                             0.3.25
jaxlib                          0.3.22
"""

import functools
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
import cv2

from tqdm import tqdm

from tapnet import tapir_model
from tapnet.utils import transforms
from tapnet.utils import viz_utils

import imageio
import json
import os

import requests
from tools.rotate import frame_rotate


# https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy

def download_checkpoints(ck_url='https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy'):
    print('Downloading the checkpoint...')
    ipadd = '127.0.0.1:7890'
    proxies = {'http': 'http://{}'.format(ipadd),
               'https': 'https://{}'.format(ipadd)}
    try:
        checkpoint = requests.get(ck_url, proxies=None).content
    except requests.exceptions.ProxyError:
        checkpoint = requests.get(ck_url, proxies=proxies).content
    except:
        print(
            'Please download the checkpoint file at "https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint'
            '.npy" and put it into the folder "tapnet/checkpoints" manually!')
        return None
    with open(os.path.abspath('tapnet/checkpoints/' + ck_url.split(r'/')[-1]), 'wb') as f:
        f.write(checkpoint)
    f.close()
    print('Completed!')
    return None


def build_online_model_init(frames, query_points):
    """Initialize query features for the query points."""
    model = tapir_model.TAPIR(use_causal_conv=True, bilinear_interp_with_depthwise_conv=False)

    feature_grids = model.get_feature_grids(frames, is_training=False)
    query_features = model.get_query_features(
        frames,
        is_training=False,
        query_points=query_points,
        feature_grids=feature_grids,
    )
    return query_features


def build_online_model_predict(frames, query_features, causal_context):
    """Compute point tracks and occlusions given frames and query points."""
    model = tapir_model.TAPIR(use_causal_conv=True, bilinear_interp_with_depthwise_conv=False)
    feature_grids = model.get_feature_grids(frames, is_training=False)
    trajectories = model.estimate_trajectories(
        frames.shape[-3:-1],
        is_training=False,
        feature_grids=feature_grids,
        query_features=query_features,
        query_points_in_video=None,
        query_chunk_size=64,
        causal_context=causal_context,
        get_causal_context=True,
    )
    causal_context = trajectories['causal_context']
    del trajectories['causal_context']
    return {k: v[-1] for k, v in trajectories.items()}, causal_context


def preprocess_frames(frames):
    """Preprocess frames to model inputs.

  Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

  Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
  """
    frames = frames.astype(jnp.float32)
    frames = frames / 255 * 2 - 1
    return frames


def postprocess_occlusions(occlusions, expected_dist):
    """Postprocess occlusions to boolean visible flag.

  Args:
    occlusions: [num_points, num_frames], [-inf, inf], np.float32

  Returns:
    visibles: [num_points, num_frames], bool
  """
    pred_occ = jax.nn.sigmoid(occlusions)
    pred_occ = 1 - (1 - pred_occ) * (1 - jax.nn.sigmoid(expected_dist))
    visibles = pred_occ < 0.5  # threshold
    return visibles


def sample_random_points(frame_max_idx, height, width, num_points):
    """Sample random points with (time, height, width) order."""
    y = jnp.random.randint(0, height, (num_points, 1))
    x = jnp.random.randint(0, width, (num_points, 1))
    t = jnp.random.randint(0, frame_max_idx + 1, (num_points, 1))
    points = jnp.concatenate((t, y, x), axis=-1).astype(jnp.int32)  # [num_points, 3]
    return points


def construct_initial_causal_state(num_points, num_resolutions):
    value_shapes = {
        "tapir/~/pips_mlp_mixer/block_1_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_1_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_2_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_2_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_3_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_3_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_4_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_4_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_5_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_5_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_6_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_6_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_7_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_7_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_8_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_8_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_9_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_9_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_10_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_10_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_11_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_11_causal_2": (1, num_points, 2, 2048),
        "tapir/~/pips_mlp_mixer/block_causal_1": (1, num_points, 2, 512),
        "tapir/~/pips_mlp_mixer/block_causal_2": (1, num_points, 2, 2048),
    }
    fake_ret = {
        k: jnp.zeros(v, dtype=jnp.float32) for k, v in value_shapes.items()
    }
    return [fake_ret] * num_resolutions * 4


def adjust_image_factor(img, contrast=1, brightness=1):
    from PIL import Image, ImageEnhance
    pil_img = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(contrast)
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(brightness)
    return np.asarray(pil_img)


if __name__ == '__main__':
    # Load the checkpoint
    print('Loading the checkpoint...')
    checkpoint_path = os.path.abspath(".") + os.sep + 'tapnet/checkpoints/causal_tapir_checkpoint.npy'
    if not os.path.exists(checkpoint_path):
        download_checkpoints()

    ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
    params, state = ckpt_state['params'], ckpt_state['state']

    # Build the model
    print('Building the model...')
    online_init = hk.transform_with_state(build_online_model_init)
    online_init_apply = jax.jit(online_init.apply)

    online_predict = hk.transform_with_state(build_online_model_predict)
    online_predict_apply = jax.jit(online_predict.apply)

    rng = jax.random.PRNGKey(42)
    online_init_apply = functools.partial(
        online_init_apply, params=params, state=state, rng=rng
    )
    online_predict_apply = functools.partial(
        online_predict_apply, params=params, state=state, rng=rng
    )

    # The default modification size is (256,256),and you'd better set it to the power of 2.
    # We found that using larger sized images as input can greatly improve accuracy.
    resize_height = 1024
    resize_width = 1024

    # Load the video
    # video_name = 'out37.mp4' #The path of the input video
    proj_dir = 'cello_1113_scale'
    video_name = r'../data/cello_1113/cello_1113_scale/video/cello_1113_21334211.avi'
    cam_num = video_name.split('_')[-1].split('.')[0]
    parent_folder = os.path.dirname(video_name)
    base_name = os.path.basename(video_name)
    file_name = proj_dir + '_' + cam_num + '_' + str(resize_height) + 'x' + str(resize_width) + '_keypoints'
    # save_sub_dir = '_'.join(base_name.split('_')[:2])
    save_folder_path = '../cello_2d_result'
    video = imageio.get_reader(os.path.abspath(video_name), 'ffmpeg')
    iter_frames = 300  # Number of iteration frames per model insertion <=video.count_frames()

    start_frame_idx = 128

    frames = None
    tracks = None
    track_result = None

    cello_keypoints = ['scroll_top', 'nut_l', 'nut_r', 'bridge_l', 'bridge_r', 'tail_gut', 'end_pin']
    bow_keypoints = ['frog', 'tip_plate']

    '''
        Labelled json file should be loaded as np.array to the variable "query_points", or you should manually set it.
        
        For example:
            If you define "n" points, you'll get an array shaped (n,3)
            3 points has been defined, and its shape is (3,3)
            then query_points seems like:
                np.array([[   0,  419, 1257],
                          [   0, 1290, 1001],
                          [   0, 2539,  994]])
            The second and the third elements of this array are positions "Y (height)" and "X (width)"  of the pixels.
    '''
    labeled_json = f'labeled_jsons/{proj_dir}_{cam_num}_{start_frame_idx}.json'
    # The path of your keypoints -> (camera_{cameraID}_{start_frame_index}). We use labelme to label them, or you can
    # use other tools to give the array of keypoints position information manually.

    if not os.path.exists(f'{save_folder_path}/{proj_dir}/{cam_num}'):
        print('Loading frames and Inferring...')
        for num in tqdm(range(video.count_frames()), desc="Loading frames"):
            if num >= start_frame_idx - 1:
                image = np.asarray(video.get_data(num), dtype=np.uint8)
                image = frame_rotate(cam_num, image)
                # image = adjust_image_factor(image, 1.5, 1.5)

                frame = media.resize_video(image[np.newaxis, :], (resize_height, resize_width))
                if frames is None:
                    height, width = image.shape[0:2]
                    frames = frame
                    if num == start_frame_idx - 1:
                        cv2.imshow('frame1', cv2.cvtColor(cv2.resize(image, (920, 1062)), cv2.COLOR_RGB2BGR))
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                else:
                    frames = np.concatenate((frames, frame), axis=0)

                if (num + 1) % iter_frames == 0 or num == video.count_frames() - 1:
                    print('\n Round [%d] is starting!' % ((num + 1) // iter_frames + int((num + 1) % iter_frames > 0)))
                    if ((num + 1) // iter_frames + int((num + 1) % iter_frames > 0)) == 1:
                        with open(labeled_json, 'r') as f:
                            labelled_info = json.load(f)
                        f.close()

                        kpdict = {}
                        for item in labelled_info['shapes']:
                            if item['label'] in cello_keypoints + bow_keypoints:
                                kpdict[(cello_keypoints + bow_keypoints).index(item['label'])] = item['points'][0]
                                continue

                        kpdict = dict(sorted(kpdict.items(), key=lambda item: item[0], reverse=False))
                        # query_points = np.concatenate((np.zeros((len(kpdict),1)),np.flip(list(kpdict.values()))),axis =1)
                        query_points = np.concatenate(
                            (np.ones((len(kpdict), 1)) * 0, np.flip(list(kpdict.values()), axis=1)), axis=1)

                        '''
                        query_points = np.array([[   0,  419, 1257],
                                                 [   0, 1290, 1001],
                                                 [   0, 2539,  994]])
                        '''

                    else:
                        '''
                        If not all frames are inserted at once, the last inference result needs to be input into the next inference.
                        '''
                        # query_points = np.concatenate((np.zeros((tracks.shape[0],1)),np.flip(tracks[:,-1,:],axis = 1)),axis =1)
                        query_points = np.concatenate(
                            (np.ones((tracks.shape[0], 1)) * (num + 1), np.flip(tracks[:, -1, :], axis=1)), axis=1)

                    query_points = transforms.convert_grid_coordinates(
                        query_points, (1, height, width), (1, resize_height, resize_width), coordinate_format='tyx')

                    query_features, _ = online_init_apply(frames=preprocess_frames(frames[None, None, 0]),
                                                          query_points=query_points[None])
                    causal_state = construct_initial_causal_state(query_points.shape[0],
                                                                  len(query_features.resolutions) - 1)

                    # Predict point tracks frame by frame
                    predictions = []
                    for i in tqdm(range(frames.shape[0]), desc="Inferring", leave=False):
                        (prediction, causal_state), _ = online_predict_apply(
                            frames=preprocess_frames(frames[None, None, i]),
                            query_features=query_features,
                            causal_context=causal_state,
                        )
                        predictions.append(prediction)

                    # Extract some information
                    tracks = np.concatenate([x['tracks'][0] for x in predictions], axis=1)
                    occlusions = np.concatenate([x['occlusion'][0] for x in predictions], axis=1)
                    expected_dist = np.concatenate([x['expected_dist'][0] for x in predictions], axis=1)

                    visibles = postprocess_occlusions(occlusions, expected_dist)

                    # Visualize sparse point tracks
                    tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (width, height))
                    if ((num + 1) // iter_frames + int((num + 1) % iter_frames > 0)) == 1:
                        tracks_result = tracks
                        visibles_result = visibles
                    else:
                        tracks_result = np.concatenate((tracks_result, tracks), axis=1)
                        visibles_result = np.concatenate((visibles_result, visibles), axis=1)
                    frames = None
                    print('-' * 80)
        print('\nCompleted!')

        # Save
        print('Save the information of keypoints...')

        # change the shape from (keypoints_num, frame_num, 2) into (frame_num, keypoints_num, 2)
        # tracks_result = np.transpose(np.asarray(tracks_result), (1, 0, 2))
        pos = np.concatenate((tracks_result, visibles_result[:, :, np.newaxis]), axis=2).transpose(1, 0, 2)

        if not os.path.exists(save_folder_path):
            os.mkdir(save_folder_path)
        save_sub_dir_path = save_folder_path + os.sep + proj_dir
        if not os.path.exists(save_sub_dir_path):
            os.mkdir(save_sub_dir_path)
        save_sub_sub_dir_path = save_sub_dir_path + os.sep + cam_num
        if not os.path.exists(save_sub_sub_dir_path):
            os.mkdir(save_sub_sub_dir_path)

        for idx, frame in enumerate(pos):
            with open(f'{save_folder_path}/{proj_dir}/{cam_num}/{start_frame_idx + idx}.json', 'w') as f:
                f.write(json.dumps(frame.tolist()))
            f.close()

        # with open(os.path.abspath(parent_folder) + os.sep + file_name + '.json', 'w') as f:
        #     f.write(json.dumps(tracks_result.tolist()))
        # f.close()
        print('Completed!')

        readfile = tracks_result

    colormap = viz_utils.get_colors(len(cello_keypoints + bow_keypoints))

    print('Generate a video...')
    plot_flag = False  # plot_flag = True -> Use matplotlib.pyplot to visualize

    if not os.path.exists(f'{proj_dir}'):
        os.mkdir(f'{proj_dir}')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_size = tuple(np.flip(video.get_data(0).shape)[1:])  # tuple->(width,height)
    out = cv2.VideoWriter(f'{proj_dir}/{file_name}.avi', fourcc, fps=30, frameSize=np.flip(frame_size))

    # Visualize and generate a video.
    for num in tqdm(range(video.count_frames())):
        if num >= start_frame_idx - 1:
            image = np.asarray(video.get_data(num), dtype=np.uint8)
            image = frame_rotate(cam_num, image)
            with open(f'{save_folder_path}/{proj_dir}/{cam_num}/{num + 1}.json', 'r') as f:
                xyloc = np.asarray(json.load(f))[:, :2]
            f.close()

            for j, color in enumerate(colormap):
                frame = cv2.circle(image,
                                   tuple(np.array(np.round(xyloc[j]),
                                                  dtype=np.int16)), 1, color, 10)
            if plot_flag:
                plt.imshow(frame)
                plt.tight_layout()
                plt.show()
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print('Completed!')
