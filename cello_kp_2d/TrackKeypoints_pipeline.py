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
import argparse
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

try:
    from tools.rotate import frame_rotate
except:
    from rotate import frame_rotate

from icecream import ic

useYOLO = False
if useYOLO:
    from ultralytics import YOLO
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)


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
    if not os.path.exists('tapnet/checkpoints/'):
        os.mkdir('tapnet/checkpoints/')
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


def cal_dist(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2)))

def YOLO_detection(img_bgr,model):

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='track_keypoints_pipeline')
    parser.add_argument('--proj_dir', default='cello_1113_scale', type=str, required=True)
    parser.add_argument('--video_path', default=r'../data/cello_1113/cello_1113_scale/video/cello_1113_21334237.avi',
                        type=str, required=True)
    parser.add_argument('--start_frame_idx', default='128', type=int, required=True)
    parser.add_argument('--instrument', default='cello', type=str, required=True)
    parser.add_argument('--end_frame_idx', default='786', type=int, required=True)
    args = parser.parse_args()
    proj_dir = args.proj_dir
    video_path = args.video_path
    start_frame_idx = args.start_frame_idx
    instrument = args.instrument
    end_frame_idx = args.end_frame_idx

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
    #proj_dir = 'tiane'
    # proj_dir = 'aidelizan'
    #video_path = r'../data/tiane/tiane_21334237.avi'
    # video_path = r'../data/cello_0111/aidelizan/videos/aidelizan_21334181.avi'
    # instrument = 'cello'
    save_folder_path = './kp_result'
    YOLO_model_path = 'best.pt'

    cam_num = video_path.split('_')[-1].split('.')[0]
    parent_folder = os.path.dirname(video_path)
    base_name = os.path.basename(video_path)
    file_name = cam_num + '_' + str(resize_height) + 'x' + str(resize_width) + '_keypoints'
    # save_sub_dir = '_'.join(base_name.split('_')[:2])

    video = imageio.get_reader(os.path.abspath(video_path),  'ffmpeg')

    iter_frames = 300  # Number of iteration frames per model insertion <=video.count_frames()
    # start_frame_idx = 181
    # end_frame_idx = video.count_frames()
    # end_frame_idx = 1689 #635

    frames = None
    tracks = None
    track_result = None

    testmode = True

    violin_keypoints = ['scroll_top', 'nut_l', 'nut_r', 'bridge_l', 'bridge_r']
    # violin_keypoints = ['scroll_top', 'nut_l', 'nut_r', 'fingerboard_bottom_l', 'fingerboard_bottom_r']
    cello_keypoints = ['scroll_top', 'nut_l', 'nut_r', 'bridge_l', 'bridge_r', 'tail_gut', 'end_pin']

    instrument_keypoints = cello_keypoints if instrument !='violin' else violin_keypoints
    #bow_keypoints = ['frog','middle','tip_plate']
    bow_keypoints = ['frog', 'tip_plate']
    # bow_keypoints = ['tip_plate']
    # bow_keypoints = []

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
    labeled_json = f'labeled_jsons/{proj_dir}/{cam_num}/{cam_num}_{start_frame_idx}.json'
    assert open(labeled_json)
    # The path of your keypoints -> (camera_{cameraID}_{start_frame_index}). We use labelme to labe l them, or you can
    # use other tools to give the array of keypoints position information manually.

    # load YOLOV8 model
    #YOLO_model = YOLO(YOLO_model_path)
    #YOLO_model.to(device)

    frame_jsonlist= []
    json_files_indir = os.listdir(os.path.dirname(labeled_json))
    frame_jsonlist.append(end_frame_idx)

    for jsonfile in json_files_indir:
        if "_".join(os.path.basename(labeled_json).split(".json")[0].split("_")[:-1]) in jsonfile:
            json_index = int(os.path.splitext(jsonfile)[0].split("_")[-1])
            if json_index == start_frame_idx:
                frame_jsonlist.append(json_index)
            else:
                frame_jsonlist.append(json_index-1)
    frame_jsonlist = sorted(list(set(frame_jsonlist)))
    frame_alllist = frame_jsonlist.copy()
    frame_jsonlist_round = 0
    ceaseflag = 0

    for i in range(len(frame_jsonlist) - 1):
        for j in range((frame_jsonlist[i + 1] - frame_jsonlist[i]) // iter_frames):
            frame_alllist.insert(frame_alllist.index(frame_jsonlist[i]) + j + 1, frame_jsonlist[i] + iter_frames * (j + 1)-1)
    frame_alllist = sorted(list(set(frame_alllist)))#[:-1]
    frame_jsonlist = sorted(list(set(frame_jsonlist)))[:-1]

    frame_ceaselist = []
    for i in range(len(frame_jsonlist)):
        if frame_jsonlist[i] in frame_alllist:
            print(frame_jsonlist[i])
            frame_ceaselist.append(frame_alllist[(frame_alllist.index(frame_jsonlist[i])+1)])
    frame_alllist = frame_alllist[1:]
    print(frame_alllist)
    print(frame_jsonlist)
    print(frame_ceaselist)

    if not os.path.exists(f'{save_folder_path}/{proj_dir}/{cam_num}'):
        print('Loading frames and Inferring...')
        for num in tqdm(range(end_frame_idx), desc="Loading frames"):
            if num >= start_frame_idx - 1:
                image = np.asarray(video.get_data(num), dtype=np.uint8)
                image = frame_rotate(cam_num, image)
                # image = adjust_image_factor(image, 1.5, 1.5)

                # YOLO_results = YOLO_model(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                # if len(YOLO_results[0].boxes.cls):
                #     # 预测框的 xyxy 坐标
                #     YOLO_bboxes_xyxy = YOLO_results[0].boxes.xyxy.cpu().numpy().astype('uint32')
                #
                #     # 关键点的 xy 坐标
                #     try:
                #         YOLO_keypoints = YOLO_results[0].keypoints.cpu().numpy().astype('uint32')[0]
                #
                #     except:
                #         YOLO_keypoints = YOLO_results[0].keypoints.data.cpu().numpy().astype('uint32')[0]
                #         YOLO_keypoints_conf = YOLO_results[0].keypoints.conf.cpu().numpy()[0]

                    # tip_bbox = YOLO_bboxes_xyxy[0]
                    # tip_position = YOLO_bboxes_keypoints[0]
                    # tip_conf = YOLO_keypoints_conf[0]
                #else:
                    #tip_bbox = (0,0)
                    #tip_position = (0,0)
                    #tip_conf = 0

                frame = media.resize_video(image[np.newaxis, :], (resize_height, resize_width))
                if frames is None:
                    height, width = image.shape[0:2]
                    frames = frame
                    if num == start_frame_idx - 1:
                        pass
                        # cv2.imshow('frame1', cv2.cvtColor(cv2.resize(image, (920, 1062)), cv2.COLOR_RGB2BGR))
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                else:
                    frames = np.concatenate((frames, frame), axis=0)

                if len(frame_alllist)>1 and (num + 1) == frame_alllist[frame_jsonlist_round+1] :  # or num == end_frame_idx - 1:
                    frame_jsonlist_round += 1

                if (num + 1) == frame_alllist[frame_jsonlist_round] :
                    print('\n Round [%d] is starting!' % (frame_jsonlist_round+1))
                    if  (num + 1) in frame_ceaselist:
                        if ceaseflag:
                            with open(os.path.dirname(labeled_json)+os.sep+
                                      '_'.join(os.path.basename(labeled_json).split(".json")[0].split("_")[:-1])
                                      + '_' + str(frame_jsonlist[ceaseflag]+1) +'.json', 'r') as f:
                                labelled_info = json.load(f)
                            f.close()
                        else:
                            with open(os.path.dirname(labeled_json)+os.sep+
                                      '_'.join(os.path.basename(labeled_json).split(".json")[0].split("_")[:-1])
                                      + '_' + str(frame_jsonlist[ceaseflag]) +'.json', 'r') as f:
                                labelled_info = json.load(f)
                            f.close()

                        ceaseflag += 1
                        kpdict = {}
                        for item in labelled_info['shapes']:
                            if item['label'] in instrument_keypoints + bow_keypoints:
                                kpdict[(instrument_keypoints + bow_keypoints).index(item['label'])] = item['points'][0]
                                continue

                        kpdict = dict(sorted(kpdict.items(), key=lambda item: item[0], reverse=False))

                        tracks_load = np.array(list(kpdict.values()))[:, :, np.newaxis].transpose(0, 2, 1)
                        visibles_load = np.ones((tracks_load.shape[0], 1),dtype=bool)

                        query_points = np.concatenate(
                            (np.ones((len(kpdict), 1)) * (num + 1), np.flip(list(kpdict.values()), axis=1)), axis=1)


                        '''
                        query_points = np.array([[   0,  419, 1257]
                                                 [   0, 1290, 1001],
                                                 [   0, 2539,  994]])
                        '''
                    else:
                        '''
                        If not all frames are inserted at once, the last inference result needs to be input into the next inference.
                        '''

                        # query_points = np.concatenate((np.zeros((tracks.shape[0],1)),np.flip(tracks[:,-1,:],axis = 1)),axis =1)#_result[:, -1, :]
                        query_points = np.concatenate(
                            (np.ones((tracks.shape[0], 1)) * (num + 1), np.flip(tracks[:,-1,:], axis=1)), axis=1)

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

                    if frame_jsonlist_round == 0:
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

        save_sub_sub_dir_path = save_folder_path + os.sep + proj_dir + os.sep + cam_num
        if not os.path.exists(save_sub_sub_dir_path):
            os.makedirs(save_sub_sub_dir_path, exist_ok=True)

        for idx, kp_info in enumerate(pos):
            try:
                with open(r'..\human_kp_2d\kp_result'+os.sep+proj_dir+os.sep+cam_num+os.sep+str(start_frame_idx + idx)+'.json','r') as f:
                    dwpose_result = np.asarray(json.loads(f.read()))
                f.close()

                frog_pos = [(dwpose_result[126 - 1][:2] + dwpose_result[122 - 1][:2]) / 2,
                            (dwpose_result[126-1][:2]+dwpose_result[127-1][:2]+dwpose_result[122-1][:2]+dwpose_result[123-1][:2])/4,
                            (dwpose_result[127-1][:2]+dwpose_result[123-1][:2])/2,
                            (dwpose_result[127-1][:2]+dwpose_result[128-1][:2]+dwpose_result[123-1][:2]+dwpose_result[124-1][:2])/4,
                            (dwpose_result[128-1][:2]+dwpose_result[124-1][:2])/2]
                if idx == 0:
                    try:
                        with open(os.path.dirname(labeled_json) + os.sep +
                                  '_'.join(os.path.basename(labeled_json).split(".json")[0].split("_")[:-1])
                                  + '_' + str(start_frame_idx) + '.json', 'r') as f:
                            labelled_info = json.load(f)
                        f.close()

                        for i in range(len(labelled_info['shapes'])):
                            if labelled_info['shapes'][i]['label'] == 'frog':
                                # print(jsfile['shapes'][i]['points'])
                                frog_labelled = labelled_info['shapes'][i]['points'][0]
                                break

                        frog_resdist = []
                        for i in range(len(frog_pos)):
                            frog_resdist.append(cal_dist(frog_labelled,frog_pos[i]))
                        '''
                        fixed place for frog
                        '''
                        option_id = 3
                        '''
                        dynamic place for frog
                        '''
                        # np.argmin(frog_resdist)
                    except:
                        option_id = 3

                kp_info = kp_info.tolist()
                kp_info.insert(len(kp_info)-1,[frog_pos[option_id][0],frog_pos[option_id][1],1.0])

                with open(f'{save_folder_path}/{proj_dir}/{cam_num}/{start_frame_idx + idx}.json', 'w') as f:
                    f.write(json.dumps(kp_info))
                f.close()
            except:
                print("not found ground truth [frog]")
                with open(f'{save_folder_path}/{proj_dir}/{cam_num}/{start_frame_idx + idx}.json', 'w') as f:
                    f.write(json.dumps(kp_info.tolist()))
                f.close()

        # with open(os.path.abspath(parent_folder) + os.sep + file_name + '.json', 'w') as f:
        #     f.write(json.dumps(tracks_result.tolist()))
        # f.close()
        print('Completed!')

        readfile = tracks_result


    #assert()
    try:
        for idx, kp_info in enumerate(range(end_frame_idx-start_frame_idx+1)):
            with open(f'{save_folder_path}/{proj_dir}/{cam_num}/{start_frame_idx + idx}.json', 'r') as f:
                TAPjsonfile = json.loads(f.read())
            f.close()

            #with open(f'{save_folder_path}/{proj_dir}/{cam_num}/{start_frame_idx + idx}.json', 'r') as f:
            with open('{}/{}/json2labelme_{}/{}.json'.format(save_folder_path,proj_dir,cam_num,
                                                             proj_dir+'_'+str(cam_num)+'_'+str(start_frame_idx + idx)), 'r') as f:
                tipplate_gt = json.loads(f.read())
            f.close()
            print(tipplate_gt['shapes'][0]['points'][0])
            TAPjsonfile[-1][:2] = tipplate_gt['shapes'][0]['points'][0]
            TAPjsonfile[-1][-1] = 1.0
            with open(f'{save_folder_path}/{proj_dir}/{cam_num}/{start_frame_idx + idx}.json', 'w') as f:
                f.write(json.dumps(TAPjsonfile))
            f.close()
    except:
        print("not found ground truth [tip_plate]")

    #f.close()
    # if
    #     for i in range(len(pos)):
    #     with open(r'human_2d_result'+os.sep+proj_dir+os.sep+cam_num+os.sep+str(si)+'.json','r') as f:
    #         dwpose_result = np.asarray(json.loads(f.read()))
    #     f.close()
    #     frog_pos = [(dwpose_result[126-1][:2]+dwpose_result[122-1][:2])/2,
    #                 (dwpose_result[126-1][:2]+dwpose_result[127-1][:2]+dwpose_result[122-1][:2]+dwpose_result[123-1][:2])/4,
    #                 (dwpose_result[127-1][:2]+dwpose_result[123-1][:2])/2,
    #                 (dwpose_result[127-1][:2]+dwpose_result[128-1][:2]+dwpose_result[123-1][:2]+dwpose_result[124-1][:2])/4,
    #                 (dwpose_result[128-1][:2]+dwpose_result[124-1][:2])/2]

    #if 'frog' in bow_keypoints

    colormap = viz_utils.get_colors(len(instrument_keypoints + bow_keypoints))

    print('Generate a video...')
    plot_flag = False  # plot_flag = True -> Use matplotlib.pyplot to visualize

    # if not os.path.exists(f'{proj_dir}'):
    #     os.mkdir(f'{proj_dir}')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_size = tuple(np.flip(video.get_data(0).shape)[1:])  # tuple->(width,height)
    out = cv2.VideoWriter(f'./kp_result/{proj_dir}/{cam_num}/{file_name}.avi', fourcc, fps=30, frameSize=np.flip(frame_size))
    # print(f'{proj_dir}/{file_name}.avi')

    det_error_frame_li = []
    # Visualize and generate a video.
    for num in tqdm(range(end_frame_idx)):
        if num >= start_frame_idx - 1:
            image = np.asarray(video.get_data(num), dtype=np.uint8)
            image = frame_rotate(cam_num, image)
            with open(f'{save_folder_path}/{proj_dir}/{cam_num}/{num + 1}.json', 'r') as f:
                TAPjsonfile= np.asarray(json.load(f))
                # TODO visualize tip_plate
                # xyloc = TAPjsonfile[:, :2][:-1]
                xyloc = TAPjsonfile[:, :2]
                #ic(np.asarray(json.load(f))[:, :2])
                acc = TAPjsonfile[:, -1]
            f.close()
            if (acc.all()==False):
                det_error_frame_li.append(num+1)
            #with open(f'../human_2d_result/toy_1226/21334237/{num + 1}.json') as f:
             #   inferdata = np.asarray(json.load(f))
              #  tip_plate = inferdata[123, :2]
            #f.close()

            if testmode:
                image = cv2.putText(image, str(num+1), (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 2)
            #if acc[-3]==0:
                #image = cv2.putText(image,'tip_occasion',(100,100),cv2.FONT_HERSHEY_SIMPLEX,5,(0,255,0),5)

            #tip_plate_temp = np.array([1059.0, 1190.0])#kpdict['tip_plate']
            #frog = np.array([1972.0, 1308.0])
            #bow_length = np.sqrt(np.sum(np.square(frog - tip_plate_temp)))

            #ic(tip_plate)
            #ic(inferdata)

            #unit_vector = (xyloc[-2]-tip_plate)/np.sqrt(np.sum(np.square(xyloc[-2] - tip_plate)))*bow_length+tip_plate
            #ic(unit_vector)
            #image = cv2.circle(image,
            #                   tuple(np.array(np.round(unit_vector),
            #                                  dtype=np.int16)), 1, (0,255,0), 10)
            #image = cv2.circle(image,
             #                  tuple(np.array(np.round(tip_plate),
              #                                dtype=np.int16)), 1, (255, 0 , 255), 10)
            for j, color in enumerate(colormap):
                # print(j)
                # print(color)
                # print(xyloc[j])
                frame = cv2.circle(image,
                                   tuple(np.array(np.round(xyloc[j]),
                                                  dtype=np.int16)), 1, color, 10)
            if plot_flag:
                plt.imshow(frame)
                plt.tight_layout()
                plt.show()

            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if det_error_frame_li:
        from array import array
        with open(f'./kp_result/{proj_dir}/{cam_num}/{file_name}_error.txt', 'w') as f:
            det_error_frame_arr = array('i', det_error_frame_li)
            f.write(','.join(map(str, det_error_frame_arr)))
        f.close()
    print('Completed!')
