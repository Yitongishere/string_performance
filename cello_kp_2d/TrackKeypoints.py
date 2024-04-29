# -*- coding: utf-8 -*-
"""
Created on Thu April 4 20:00:00 2024

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

#libraries for basic functions
import json
import os
from tqdm import tqdm
import sys
sys.path.append('..')

#libraries for computing
import numpy as np
import math

#libraries for processing images
import cv2
import imageio
import mediapy as media
import matplotlib.pyplot as plt
from tools.rotate import frame_rotate

# use network to download checkpoints
import requests

#verbose
#from icecream import ic

#libraries of tapnet
import jax
from tapnet import tapir_model
from tapnet.utils import transforms
from tapnet.utils import viz_utils
from tapnet.utils import model_utils

#libraries of YOLOv8
from ultralytics import YOLO
import torch

#libraries of Deeplsd
from deeplsd.utils.tensor import batch_to_device
from deeplsd.models.deeplsd_inference import DeepLSD
from deeplsd.geometry.viz_2d import plot_images, plot_lines


######################## Download checkpoints ########################
def download_checkpoints(ckpt_url,ckpt_path):
    print('Downloading the checkpoint...\nfrom "{}"'.format(ckpt_url))
    IP_address = '127.0.0.1:7890'
    proxies = {'http': 'http://{}'.format(IP_address),
               'https': 'https://{}'.format(IP_address)}
    try:
        checkpoint_file = requests.get(ckpt_url, proxies=None).content
    except requests.exceptions.ProxyError:
        checkpoint_file = requests.get(ckpt_url, proxies=proxies).content
    except:
        return False
    if not os.path.exists(os.path.dirname(ckpt_path)):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    with open(os.path.abspath(os.path.dirname(ckpt_path)+ os.sep + ckpt_url.split(r'/')[-1]), 'wb') as f:
        f.write(checkpoint_file)
    f.close()
    print('Completed!')
    return True

    
def DeepLSD_download_checkpoint(summary):
    DeepLSD_model_type = summary['DeepLSD_model_type']
    ckpt_path = summary['DeepLSD_ckpt_path']
    ckpt_urls = {'md':'https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_md.tar',# Trained by MegaDepth Dataset
                 'wireframe':'https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_wireframe.tar',# Trained by Wireframe Dataset
                }
    ckpt_url = ckpt_urls[DeepLSD_model_type]
    #ckpt_url = ckpt_urls['md']
    #ckpt_path += ckpt_url.split('/')[-1]
    if not download_checkpoints(ckpt_url,ckpt_path):
        requests.exceptions.ConnectTimeout(
            'Please download the checkpoint file at {} and'
            'put it into the folder "DeepLSD/checkpoints/" manually!\n'
            'You can choose one from below links:\n{}'.format(ckpt_url,ckpt_urls))
    return None
######################## Download checkpoints ########################



################################# TAPIR #################################
def TAPIR_download_checkpoint(summary):
    MODEL_TYPE = summary['TAPIR_model_type']
    ckpt_path = summary['TAPIR_ckpt_path']
    ckpt_urls = {'tapnet':'https://storage.googleapis.com/dm-tapnet/checkpoint.npy',# TAP-Net
                 'tapir':'https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.npy',# TAPIR
                 'causal':'https://storage.googleapis.com/dm-tapnet/causal_tapir_checkpoint.npy',# Online TAPIR
                 'bootstapir':'https://storage.googleapis.com/dm-tapnet/bootstapir_checkpoint.npy',# BootsTAPIR
                }
    ckpt_url = ckpt_urls['TAPIR_model_type']
    
    if not download_checkpoints(ckpt_url,ckpt_path):
        requests.exceptions.ConnectTimeout(
            'Please download the checkpoint file at {} and'
            'put it into the folder "tapnet/checkpoints" manually!\n'
            'You can choose one from below links:\n{}'.format(ckpt_url,ckpt_urls))
    return None


def TAPIR_test_checkpoint(summary):
    def online_model_init(frames, query_points):
        """Initialize query features for the query points."""
        frames = model_utils.preprocess_frames(frames)
        feature_grids = tapir.get_feature_grids(frames, is_training=False)
        query_features = tapir.get_query_features(
          frames,
          is_training=False,
          query_points=query_points,
          feature_grids=feature_grids,
        )
        return query_features
    
    def online_model_predict(frames, query_features, causal_context):
        """Compute point tracks and occlusions given frames and query points."""
        frames = model_utils.preprocess_frames(frames)
        feature_grids = tapir.get_feature_grids(frames, is_training=False)
        trajectories = tapir.estimate_trajectories(
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
        # Take only the predictions for the final resolution.
        # For running on higher resolution, it's typically better to average across
        # resolutions.
        tracks = trajectories['tracks'][-1]
        occlusions = trajectories['occlusion'][-1]
        uncertainty = trajectories['expected_dist'][-1]
        visibles = model_utils.postprocess_occlusions(occlusions, uncertainty, 0.9)
        return tracks, visibles, causal_context
    
    def inference(frames, query_points):
        """
            Inference on one video.

            Args:
            frames: [num_frames, height, width, 3], [0, 255], np.uint8
            query_points: [num_points, 3], [0, num_frames/height/width], [t, y, x]

            Returns:
            tracks: [num_points, 3], [-1, 1], [t, y, x]
            visibles: [num_points, num_frames], bool
        """
        # Preprocess video to match model inputs format
        frames = model_utils.preprocess_frames(frames)
        query_points = query_points.astype(np.float32)
        frames, query_points = frames[None], query_points[None]  # Add batch dimension

        outputs = tapir(video=frames, query_points=query_points, is_training=False, query_chunk_size=32)
        tracks, occlusions, expected_dist = outputs['tracks'], outputs['occlusion'], outputs['expected_dist']

        # Binarize occlusions
        visibles = model_utils.postprocess_occlusions(occlusions, expected_dist, 0.9)
        return tracks[0], visibles[0]
    
    
    MODEL_TYPE = summary['TAPIR_model_type']
    ck_path = summary['TAPIR_ckpt_path']
    ckpt_state = np.load(ck_path, allow_pickle=True).item()
    params, state = ckpt_state['params'], ckpt_state['state']
    if MODEL_TYPE == 'causal':
        kwargs = dict(use_causal_conv=True, bilinear_interp_with_depthwise_conv=False)
    elif MODEL_TYPE == 'tapir':
        kwargs = dict(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
    elif MODEL_TYPE == 'bootstapir':
        kwargs = dict(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
        kwargs.update(dict(
        pyramid_level=1,
        extra_convs=True,
        softmax_temperature=10.0
      ))
    else:
        pass
    
    tapir = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=kwargs)
    if MODEL_TYPE == 'causal':
        online_model_init= jax.jit(online_model_init)
        online_model_predict = jax.jit(online_model_predict)
        return tapir, online_model_init, online_model_predict
    else:
        inference = jax.jit(inference)
        return tapir, inference


def get_seperate_list(summary):
    start_frame_idx = summary['start_frame_idx']
    end_frame_idx = summary['end_frame_idx']
    iter_frames = summary['iter_frames']
    labeled_json = summary['labeled_json']
    
    
    frame_jsonlist= []
    json_files_indir = os.listdir(os.path.dirname(labeled_json))
    frame_jsonlist.append(end_frame_idx)

    for jsonfile in json_files_indir:
        if "_".join(os.path.basename(labeled_json).split(".json")[0].split("_")[:-1]) in jsonfile:
            json_index = int(os.path.splitext(jsonfile)[0].split("_")[-1])
            #if json_index == start_frame_idx:
            frame_jsonlist.append(json_index)
            #else:
                #frame_jsonlist.append(json_index-1)


    frame_jsonlist = sorted(list(set(frame_jsonlist)))
    frame_alllist = frame_jsonlist.copy()
    frame_jsonlist_round = 0
    ceaseflag = 0

    for i in range(len(frame_jsonlist) - 1):
        for j in range((frame_jsonlist[i + 1] - frame_jsonlist[i]) // (iter_frames)):
            frame_alllist.insert(frame_alllist.index(frame_jsonlist[i]) + j + 1, frame_jsonlist[i] + (iter_frames-1) * (j + 1))
    frame_alllist = sorted(list(set(frame_alllist)))#[:-1]
    frame_jsonlist = sorted(list(set(frame_jsonlist)))[:-1]

    frame_cyclelist = []
    for i in range(len(frame_jsonlist)):
        if frame_jsonlist[i] in frame_alllist:
            frame_cyclelist.append(frame_alllist[(frame_alllist.index(frame_jsonlist[i])+1)])
    frame_alllist = frame_alllist[1:]
    
    summary.update(var_to_dict(frame_jsonlist = frame_jsonlist))
    summary.update(var_to_dict(frame_alllist = frame_alllist))
    summary.update(var_to_dict(frame_cyclelist = frame_cyclelist))
    
    return summary


def get_origin(summary):
    ROI_size = summary['ROI_size']
    resize_pixel = summary['resize_pixel']
    keypoints = summary['instrument_kps']
    labeled_json = summary['labeled_json']
    frame_jsonlist = summary['frame_jsonlist']
    ceaseflag = 0
    
    with open(os.path.dirname(labeled_json)+os.sep+
          '_'.join(os.path.basename(labeled_json).split(".json")[0].split("_")[:-1])
          + '_' + str(frame_jsonlist[ceaseflag]) +'.json', 'r') as f:
        labelled_info = json.load(f)
    f.close()
    
    kpdict = {}
    for item in labelled_info['shapes']:
        if item['label'] in keypoints:
            kpdict[(keypoints).index(item['label'])] = item['points'][0]
            continue

    kpdict = dict(sorted(kpdict.items(), key=lambda item: item[0], reverse=False))
    centerpoints = np.mean(list(kpdict.values()),axis = 0,dtype = np.int16)
    
    origin = centerpoints-int(ROI_size/2)
    summary.update(origin = origin)
    return summary


def TAPIR_infer(summary):
    start_frame_idx = summary['start_frame_idx']
    end_frame_idx = summary['end_frame_idx']
    iter_frames = summary['iter_frames']
    labeled_json = summary['labeled_json']
    origin = summary['origin']
    
    video_path = summary['video_path']
    cam_num = summary['cam_num']             
    ROI_size = summary['ROI_size']
    resize_pixel = summary['resize_pixel']
    
    frame_jsonlist = summary['frame_jsonlist']
    frame_alllist = summary['frame_alllist']
    frame_cyclelist = summary['frame_cyclelist']
    
    MODEL_TYPE = summary['TAPIR_model_type']
    TAPIR_model = summary['TAPIR_model']
    '''
    # Build the model
    print('Building the model...')
    if MODEL_TYPE == 'causal':
        TAPIR_model, TAPIR_online_model_init, TAPIR_online_model_predict = TAPIR_test_checkpoint(summary)
    else:
        TAPIR_model, TAPIR_inference = TAPIR_test_checkpoint(summary)
    '''
    if MODEL_TYPE == 'causal':
        TAPIR_online_model_init = summary['TAPIR_online_model_init']
        TAPIR_online_model_predict = summary['TAPIR_online_model_predict']
    else:
        TAPIR_inference = summary['TAPIR_inference']
    
    
    instrument_kps = summary['instrument_kps']
    guided_kps = summary['guided_kps']
    keypoints = instrument_kps + guided_kps
    
    video = imageio.get_reader(os.path.abspath(video_path),  'ffmpeg')
    

    frames = None
    frame_jsonlist_round = 0
    ceaseflag = 0
    
    for num in tqdm(range(end_frame_idx), desc=f"(TAP:{MODEL_TYPE}) Loading frames"):
        if (num + 1) >= start_frame_idx:
            image = np.asarray(video.get_data(num), dtype=np.uint8)
            image = frame_rotate(cam_num, image) 
            image = image[origin[1]:origin[1]+ROI_size,origin[0]:origin[0]+ROI_size,:]
            #image = remove_flash(image)
            frame = media.resize_video(image[np.newaxis, :], (resize_pixel, resize_pixel))

            if frames is None:
                height, width = image.shape[0:2]
                frames = frame
                if num + 1 == start_frame_idx:
                    pass
            else:
                frames = np.concatenate((frames, frame), axis=0)

            if len(frame_alllist)>1 and (num + 1) == frame_alllist[frame_jsonlist_round+1] :
                frame_jsonlist_round += 1

            if (num + 1) == frame_alllist[frame_jsonlist_round] :
                print('\n Round [%d] is starting!' % (frame_jsonlist_round+1))
                #if (num - iter_frames + 2) in frame_jsonlist or len(frame_alllist) == 1:
                if (num - iter_frames + 2) in frame_jsonlist or (num + 1) == frame_jsonlist[min(ceaseflag+1,len(frame_jsonlist)-1)] or len(frame_alllist) == 1:
                    with open(os.path.dirname(labeled_json)+os.sep+
                              '_'.join(os.path.basename(labeled_json).split(".json")[0].split("_")[:-1])
                              + '_' + str(frame_jsonlist[ceaseflag]) +'.json', 'r') as f:
                        labelled_info = json.load(f)

                    ceaseflag += 1
                    kpdict = {}
                    for item in labelled_info['shapes']:
                        if item['label'] in keypoints:
                            kpdict[(keypoints).index(item['label'])] = item['points'][0]
                            continue

                    kpdict = dict(sorted(kpdict.items(), key=lambda item: item[0], reverse=False))

                    query_points = np.concatenate(
                        (np.ones((len(kpdict), 1)) * (num - iter_frames + 1), np.flip(list(kpdict.values())-origin, axis=1)), axis=1)
                    #print('qp_ori:',query_points)

                else:
                    query_points = np.concatenate(
                        (np.ones((tracks.shape[0], 1)) * (num - iter_frames + 2), np.flip(tracks[:,-1,:], axis=1)), axis=1)

                query_points = transforms.convert_grid_coordinates(
                                    query_points, (1, height, width), (1, resize_pixel, resize_pixel), coordinate_format='tyx')
                
                if MODEL_TYPE == 'causal':
                    query_features = TAPIR_online_model_init(frames[None, 0:1], query_points[None])
                    causal_state = TAPIR_model.construct_initial_causal_state(query_points.shape[0], len(query_features.resolutions) - 1)
                    # Predict point tracks frame by frame
                    predictions = []
                    for infer_idx in tqdm(range(frames.shape[0]), desc="(TAP_causal) Inferring:", leave=False):
                      # Note: we add a batch dimension.
                      tracks, visibles, causal_state = TAPIR_online_model_predict(
                          frames=frames[None, infer_idx:infer_idx+1],
                          query_features=query_features,
                          causal_context=causal_state,
                      )
                      predictions.append({'tracks':tracks, 'visibles':visibles})

                    tracks = np.concatenate([x['tracks'][0] for x in predictions], axis=1)
                    visibles = np.concatenate([x['visibles'][0] for x in predictions], axis=1)
                else:
                    for _ in tqdm(range(1), desc="(TAP) Inferring:", leave=False):
                        tracks, visibles = TAPIR_inference(frames, query_points)
                        tracks = np.array(tracks)
                        visibles = np.array(visibles)

                '''
                frames_tensor = torch.tensor(frames).to(TAPIR_device)
                query_points_tensor = torch.tensor(query_points).to(TAPIR_device)
                tracks, visibles = TAPIR_inference(frames_tensor, query_points_tensor, TAPIR_model)

                tracks = tracks.cpu().numpy()
                visibles = visibles.cpu().numpy()
                '''

                # Visualize sparse point tracks    
                diff_qp = tracks.transpose(1,2,0)[0].T-np.flip(query_points[:,1:],axis = 1)
                tracks= tracks - diff_qp[:,np.newaxis,:]
                tracks = transforms.convert_grid_coordinates(tracks, (resize_pixel, resize_pixel), (height, width))
                for keypoints_num in range(visibles.shape[0]):
                    if not np.alltrue(visibles[keypoints_num]):
                        false_indices = np.squeeze(np.where(visibles[keypoints_num] == False),axis = 0)
                        for i in range(len(false_indices)):
                            dist_li = []
                            for j in range(visibles.shape[0]):
                                dist = cal_dist(tracks[keypoints_num][false_indices[i]], tracks[j][false_indices[i]])
                                dist_li.append(dist if dist != 0 else np.inf)
                            dist_argmin = np.argmin(dist_li)
                            if visibles[dist_argmin][false_indices[i] - 1] == True and visibles[dist_argmin][false_indices[i]] == True:
                                diff_guided = tracks[dist_argmin][false_indices[i]] - tracks[dist_argmin][false_indices[i] - 1]
                            else:
                                diff_guided = tracks[-1][false_indices[i]] - tracks[-1][false_indices[i] - 1]
                            tracks[keypoints_num][false_indices[i]] = tracks[keypoints_num][false_indices[i]-1] + diff_guided
                            visibles[keypoints_num][false_indices[i]] = True


                if len(frame_alllist) == 1:
                    tracks_result = tracks
                    visibles_result = visibles
                elif frame_jsonlist_round == 0:
                    tracks_result = tracks[:,:-1,:]
                    visibles_result = visibles[:,:-1]
                elif num + 1 == end_frame_idx:
                    tracks_result = np.concatenate((tracks_result, tracks), axis=1)
                    visibles_result = np.concatenate((visibles_result, visibles), axis=1)
                else:
                    tracks_result = np.concatenate((tracks_result, tracks[:,:-1,:]), axis=1)
                    visibles_result = np.concatenate((visibles_result, visibles[:,:-1]), axis=1)
                frames = media.resize_video(frames[-1][np.newaxis, :], (resize_pixel, resize_pixel))
                
                print('-' * 80)
    
    summary.update(instrument_kp_tracks = tracks_result[:len(instrument_kps),:])
    summary.update(guided_kp_tracks = tracks_result[len(instrument_kps):,:])
    return summary
################################# TAPIR #################################



################################# YOLOv8 #################################
def YOLO_detectbow_download_checkpoint(ckpt_path = './yolov8/checkpoints'):
    ckpt_url = ''
    if not download_checkpoints(ckpt_url,ckpt_path):
        requests.exceptions.ConnectTimeout(
            'Please download the checkpoint file at {} and'
            'put it into the folder "yolov8/checkpoints/" manually!'
            .format(ckpt_url))
    return None
################################# YOLOv8 #################################



################################# DeepLSD #################################
def DeepLSD_download_checkpoint(summary):#ckpt_type = 'md',ckpt_path = './DeepLSD/checkpoints'
    DeepLSD_model_type = summary['DeepLSD_model_type']
    ckpt_path = summary['DeepLSD_ckpt_path']
    ckpt_urls = {'md':'https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_md.tar',# Trained by MegaDepth Dataset
                 'wireframe':'https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_wireframe.tar',# Trained by Wireframe Dataset
                }
    ckpt_url = ckpt_urls[DeepLSD_model_type]
    #ckpt_url = ckpt_urls['md']
    #ckpt_path += ckpt_url.split('/')[-1]
    if not download_checkpoints(ckpt_url,ckpt_path):
        requests.exceptions.ConnectTimeout(
            'Please download the checkpoint file at {} and'
            'put it into the folder "deeplsd/checkpoints/" manually!\n'
            'You can choose one from below links:\n{}'.format(ckpt_url,ckpt_urls))
    return None


def DeepLSD_test_checkpoint(summary):
    torch_device = summary['torch_device']             
    ckpt_path = summary['DeepLSD_ckpt_path']
    
    ckpt_state = torch.load(ckpt_path, map_location = torch_device)             
    conf = {
            'detect_lines': True,  # Whether to detect lines or only DF/AF
            'line_detection_params': 
                {
                    'merge': True,  # Whether to merge close-by lines
                    'filtering': True,
                    # Whether to filter out lines based on the DF/AF. Use 'strict' to get an even stricter filtering
                    'grad_thresh': 3,
                    'grad_nfa': True,
                    # If True, use the image gradient and the NFA score of LSD to further threshold lines.
                    #We recommand using it for easy images, but to turn it off for challenging images (e.g. night, foggy, blurry images)
                 }
            }
    model = DeepLSD(conf)
    model.load_state_dict(ckpt_state['model'])
    model.to(torch_device)
    return model


def DeepLSD_infer(summary):
    torch_device = summary['torch_device']
    img = summary['DeepLSD_infer_image']
    model = summary['DeepLSD_model']
    # Detect (and optionally refine)  the lines
    if img.ndim == 1:
        gray_img = img.copy()
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
    else:
        raise cv2.error("Please check the input image format. The number of image channels should be 1 (GRAY) or 3 (RGB).")
    inputs = {'image': torch.tensor(gray_img, dtype=torch.float, device = torch_device)[None, None] / 255.}
    with torch.no_grad():
        out = model(inputs)
        pred_lines = out['lines'][0]
    return pred_lines
################################# DeepLSD #################################



##################### Some methods for Processing Keypoints on Bow #####################
def compute_line_length(line):
    return math.dist(line[0], line[1])

def get_neighborhood_average(image, x, y, radius):
    x_start = max(0, x - radius)
    y_start = max(0, y - radius)
    x_end = min(image.shape[1] - 1, x + radius)
    y_end = min(image.shape[0] - 1, y + radius)
    
    neighborhood = image[y_start:y_end, x_start:x_end]
    
    average_color = np.mean(neighborhood, axis=(0, 1))
    return average_color


def cal_dist(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2)))


def distance_to_boundary(x, y, width, height):
    return min(x, y, width - x, height - y)


def line_equation_two_points(x1, y1, x2, y2):
    if x2 - x1 != 0:
        k = (y2 - y1) / (x2 - x1)
    else:
        k = float('inf') 
    b = y1 - k * x1
    return k,b


def distance_point_to_line(x, y, A, B=-1, C=0):
    numerator = abs(A * x + B * y + C)
    denominator = math.sqrt(A**2 + B**2)
    distance = numerator / denominator
    return distance


def compute_longest_line(summary,pred_lines):
    instrument = summary['instrument']
    DeepLSD_infer_image = summary['DeepLSD_infer_image']
    x1,y1,x2,y2 = summary['bow_bbox'].xyxy.numpy().flatten().astype('uint32')
    
    bbox_k,bbox_b = line_equation_two_points(x1, y1, x2, y2)
    pred_lines_previous = pred_lines.copy()
    longest_line = np.array([[], []])
    
    if instrument == 'cello':
        for line in pred_lines:
            condition1 = compute_line_length(line)>max(DeepLSD_infer_image.shape[:2])/4*3
            line_k,line_b = line_equation_two_points(line[0][0], line[0][1], line[1][0],line[1][1])
            condition2 = abs(line_k) < abs(bbox_k) + 0.15 and abs(line_k) > abs(bbox_k) - 0.15

            try:
                border = 10 #min(DeepLSD_infer_image.shape[:2])/8
                radius = 3
                condition3 = distance_to_boundary(int(line[0][0]), int(line[0][1]),DeepLSD_infer_image.shape[1],DeepLSD_infer_image.shape[0])<border
                condition4 = distance_to_boundary(int(line[1][0]), int(line[1][1]),DeepLSD_infer_image.shape[1],DeepLSD_infer_image.shape[0])<border
                condition5 = np.alltrue(get_neighborhood_average(DeepLSD_infer_image, int(line[0][0]), int(line[0][1]), radius)>(100,60,50))
                condition6 = np.alltrue(get_neighborhood_average(DeepLSD_infer_image, int(line[1][0]), int(line[1][1]), radius)>(100,60,50))

            except:
                continue
            if  condition1 or (condition2 and (not (condition3 and condition4))) or (condition2 and condition5 and condition6):
                if not longest_line.any() and (condition5 and condition6) and (not (condition3 and condition4) or condition1):
                    longest_line = line
                    continue
                longest_line_length = compute_line_length(longest_line)
                current_line_length = compute_line_length(line)
                if current_line_length > longest_line_length and (condition5 and condition6) and (not (condition3 and condition4) or condition1):
                    longest_line = line
            else:
                indices = np.where(pred_lines==line)
                pred_lines = np.delete(pred_lines,indices[0],axis = 0)

        if pred_lines.shape[0] < 2 or not longest_line.any():
            pred_lines = pred_lines_previous.copy()
            longest_line = np.array([[], []])
            for line in pred_lines:
                condition1 = compute_line_length(line)>max(DeepLSD_infer_image.shape[:2])/2
                line_k,line_b = line_equation_two_points(line[0][0], line[0][1], line[1][0],line[1][1])
                condition2 = abs(line_k) < abs(bbox_k) + 0.15 and abs(line_k) > abs(bbox_k) - 0.15
                if condition1 or condition2:
                    if not longest_line.any():
                        longest_line = line
                        continue
                    longest_line_length = compute_line_length(longest_line)
                    current_line_length = compute_line_length(line)
                    if current_line_length > longest_line_length:
                        longest_line = line
                else:
                    indices = np.where(pred_lines==line)
                    pred_lines = np.delete(pred_lines,indices[0],axis = 0)
    
    else:
        for line in pred_lines:
            condition1 = compute_line_length(line)>max(DeepLSD_infer_image.shape[:2])/2
            line_k,line_b = line_equation_two_points(line[0][0], line[0][1], line[1][0],line[1][1])
            condition2 = abs(line_k) < abs(bbox_k) + 0.15 and abs(line_k) > abs(bbox_k) - 0.15
            if condition1 or condition2:
                if not longest_line.any():
                    longest_line = line
                    continue
                longest_line_length = compute_line_length(longest_line)
                current_line_length = compute_line_length(line)
                if current_line_length > longest_line_length:
                    longest_line = line
            else:
                indices = np.where(pred_lines==line)
                pred_lines = np.delete(pred_lines,indices[0],axis = 0)
    
    return longest_line,pred_lines


def cal_angle(p1,p2):
    dy1 = p2[1]-p1[1]
    dx1 = p2[0]-p1[0]
    angle = math.atan2(dy1, dx1)
    angle = int(round(angle * 180 / math.pi))
    if angle < 0:
        angle += 180
    return angle


def verify_handpos(summary,longest_line,video_num):
    proj_dir = summary['proj_dir']
    cam_num = summary['cam_num']
    x1,y1,x2,y2 = summary['bow_bbox'].xyxy.numpy().flatten().astype('uint32')
    infer_image = summary['DeepLSD_infer_image']
    longest_line_k,longest_line_b = line_equation_two_points(longest_line[0][0], longest_line[0][1],
                                                             longest_line[1][0],longest_line[1][1])

    if longest_line_k>=0:
        handpos = (0,0)
    else:
        handpos = (0,y2-y1)
    return handpos


def detect_frog_tip(summary,img,handpos,longest_line):
    x1,y1,x2,y2 = summary['bow_bbox'].xyxy.numpy().flatten().astype('uint32')
    imgbow = img[y1:y2,x1:x2,:]
    
    '''
    plot_images([imgbow], ['longest_line'], cmaps='gray')
    plotline = longest_line.reshape(1, 2, 2)
    plot_lines([plotline], line_colors='red', indices=range(1))
    plt.show()
    '''

    if math.dist(handpos,longest_line[0])<math.dist(handpos,longest_line[1]):
        frog = longest_line[0]
        tip = longest_line[1]
    else:
        frog = longest_line[1]
        tip = longest_line[0]

    
    longest_line_angle = cal_angle(longest_line[0],longest_line[1])

    aim_angle = longest_line_angle
    longest_line_k,longest_line_b = line_equation_two_points(longest_line[0][0], longest_line[0][1],
                                                            longest_line[1][0],longest_line[1][1])
    
    # 检测弓根
    frog_dis = np.inf
    frog_res = frog
    frog_det = frog
    
    while True:
        frog_li = []
        for i in range(len(pred_lines)):
            pred_line_angle = cal_angle(pred_lines[i][0],pred_lines[i][1])
            sep_points_angle = cal_angle(frog,pred_lines[i][np.argmax(pred_lines[i].copy().T[0])])

            condition1_angle_threshold = 10
            condition2_angle_threshold = 5
            condition3_distance_threshold = 5

            if aim_angle - condition1_angle_threshold < 0:
                condition1 = pred_line_angle < aim_angle + condition1_angle_threshold or pred_line_angle > 180 - aim_angle - condition1_angle_threshold
            elif aim_angle + condition1_angle_threshold > 180:
                condition1 = pred_line_angle > aim_angle - condition1_angle_threshold or pred_line_angle < 180 - aim_angle + condition1_angle_threshold
            else:    
                condition1 = aim_angle - condition1_angle_threshold < pred_line_angle < aim_angle + condition1_angle_threshold

            if aim_angle - condition2_angle_threshold < 0:
                condition2 = sep_points_angle < aim_angle + condition2_angle_threshold or sep_points_angle > 180 - aim_angle - condition2_angle_threshold
            elif aim_angle + condition2_angle_threshold > 180:
                condition2 = sep_points_angle > aim_angle - condition2_angle_threshold or sep_points_angle < 180 - aim_angle + condition2_angle_threshold
            else:
                condition2 = aim_angle - condition2_angle_threshold < sep_points_angle < aim_angle + condition2_angle_threshold

            point1_distance2longestline = distance_point_to_line(pred_lines[i][0][0],pred_lines[i][0][1],
                                                                longest_line_k,-1,longest_line_b)
            point2_distance2longestline = distance_point_to_line(pred_lines[i][1][0],pred_lines[i][1][1],
                                                                longest_line_k,-1,longest_line_b)
            condition3 = point1_distance2longestline < condition3_distance_threshold and point2_distance2longestline < condition3_distance_threshold


            if imgbow.shape[0]<imgbow.shape[1]:
                condition4 = max(pred_lines[i][0][0],pred_lines[i][1][0]) < frog_det[0]
            else:
                condition4 = max(pred_lines[i][0][1],pred_lines[i][1][1]) > frog_det[1]

            if handpos == (0,0):
                condition5 = pred_lines[i][np.argmin(pred_lines[i].copy().T[0])][1] < imgbow.shape[0] / 4
            elif handpos == (0,imgbow.shape[0]):
                condition5 = pred_lines[i][np.argmin(pred_lines[i].copy().T[0])][1] > imgbow.shape[0] / 4 * 3
            else:
                condition5 = True
            
            if condition1 and condition2 and condition3 and condition4 and condition5:
                if math.dist(frog,pred_lines[i][np.argmax(pred_lines[i].copy().T[0])])< frog_dis:
                    frog_dis = math.dist(frog,pred_lines[i][np.argmax(pred_lines[i].copy().T[0])])
                    frog_res = pred_lines[i][np.argmin(pred_lines[i].copy().T[0])]
                '''
                plot_images([imgbow], ['DeepLSD lines[frog]'], cmaps='gray')
                plotline = pred_lines[i].reshape(1, 2, 2)
                plot_lines([plotline], line_colors='red', indices=range(1))
                plt.show()
                '''
        
        frog_det = frog_res
        if frog_li == []:
            break
        else:
            aim_angle = cal_angle(frog_li[-1][0],frog_li[-1][1]) 
    
    
    # 检测弓头 
    tip_dis = np.inf
    tip_res = tip
    tip_det = tip

    aim_angle = longest_line_angle

    while True:
        tip_li = []
        for i in range(len(pred_lines)):
            pred_line_angle = cal_angle(pred_lines[i][0],pred_lines[i][1])
            sep_points_angle = cal_angle(tip,pred_lines[i][np.argmax(pred_lines[i].copy().T[0])])

            condition1_angle_threshold = 10 
            condition2_angle_threshold = 5
            condition3_distance_threshold = 5

            if aim_angle - condition1_angle_threshold < 0:
                condition1 = pred_line_angle < aim_angle + condition1_angle_threshold or pred_line_angle > 180 - aim_angle - condition1_angle_threshold
            elif aim_angle + condition1_angle_threshold > 180:
                condition1 = pred_line_angle > aim_angle - condition1_angle_threshold or pred_line_angle < 180 - aim_angle + condition1_angle_threshold
            else:    
                condition1 = aim_angle - condition1_angle_threshold < pred_line_angle < aim_angle + condition1_angle_threshold

            if aim_angle - condition2_angle_threshold < 0:
                condition2 = sep_points_angle < aim_angle + condition2_angle_threshold or sep_points_angle > 180 - aim_angle - condition2_angle_threshold
            elif aim_angle + condition2_angle_threshold > 180:
                condition2 = sep_points_angle > aim_angle - condition2_angle_threshold or sep_points_angle < 180 - aim_angle + condition2_angle_threshold
            else:    
                condition2 = aim_angle - condition2_angle_threshold < sep_points_angle < aim_angle + condition2_angle_threshold

            point1_distance2longestline = distance_point_to_line(pred_lines[i][0][0],pred_lines[i][0][1],
                                                                longest_line_k,-1,longest_line_b)
            point2_distance2longestline = distance_point_to_line(pred_lines[i][1][0],pred_lines[i][1][1],
                                                                longest_line_k,-1,longest_line_b)
            condition3 = point1_distance2longestline < condition3_distance_threshold and point2_distance2longestline < condition3_distance_threshold

            if imgbow.shape[0] < imgbow.shape[1]:
                condition4 = min(pred_lines[i][0][0],pred_lines[i][1][0]) > tip_det[0]
            else:
                condition4 = min(pred_lines[i][0][1],pred_lines[i][1][1]) < tip_det[1]

            if handpos == (0,0):
                condition5 = pred_lines[i][np.argmax(pred_lines[i].copy().T[0])][1] > imgbow.shape[0] / 4 * 3
            elif handpos == (0,imgbow.shape[0]):#imgbow.shape[0]
                condition5 = pred_lines[i][np.argmax(pred_lines[i].copy().T[0])][1] < imgbow.shape[0] / 4 
            else:
                condition5 = True

            if condition1 and condition2 and condition3 and condition4 and condition5:
                if math.dist(tip_det,pred_lines[i][np.argmin(pred_lines[i].copy().T[0])])< tip_dis :
                    tip_dis = math.dist(tip,pred_lines[i][np.argmin(pred_lines[i].copy().T[0])])
                    tip_res = pred_lines[i][np.argmax(pred_lines[i].copy().T[0])]
                    tip_li.append(pred_lines[i])
                    
                    '''
                    plot_images([imgbow], ['DeepLSD lines[tip]'], cmaps='gray')
                    plotline = pred_lines[i].reshape(1, 2, 2)
                    plot_lines([plotline], line_colors='red', indices=range(1))
                    plt.show()
                    '''
                    
        tip_det = tip_res
        if tip_li == []:
            break
        else:
            aim_angle = cal_angle(tip_li[-1][0],tip_li[-1][1])

    return frog_res,tip_res


def improved_frog_tip(summary,video_num,frog,tip,handpos,image,previous_frog_id = None):
    instrument = summary['instrument']
    proj_dir = summary['proj_dir']
    cam_num = summary['cam_num']
    bow_bbox = summary['bow_bbox']
    x1,y1,x2,y2 = bow_bbox.xyxy.numpy().flatten().astype('uint32')#summary['bow_bbox_xyxy'].astype('uint32')
    conf = inform['bow_bbox'].conf.float()[0] if inform['bow_bbox'].conf.float()[0] >= inform['YOLO_conf_threshhold'] else 0
    
    detected_line_angle = cal_angle(frog,tip)

    bow_k,bow_b = line_equation_two_points(frog[0], frog[1],
                                            tip[0],  tip[1])
    bbox_k,bbox_b = line_equation_two_points(x1, y1,
                                             x2, y2)

    #imporve tip
    tip += (x1,y1)
    previous_tip = tip.copy()
    condition1 = abs(bow_k)<=0.1
    angle_threshhold = 10
    
    if handpos == (0,0): 
        condition2 = math.dist((x2,y2),tip)>math.dist((x1,y1),(x2,y2))/3
    elif handpos == (0,y2-y1):
        condition2 = math.dist((x2,0),tip)>math.dist((x1,y1),(x2,y2))/3
    condition3 = abs(detected_line_angle-cal_angle(tip,(x2,y2))) <= angle_threshhold or abs(detected_line_angle-cal_angle(tip,(x2,y2))) >= 180 - angle_threshhold
    
    if conf > 0:
        if condition1 or (bow_k < -0.1 and instrument == 'cello'):
            tip -= (x1,y1)
            tip[1] = bow_k*(x2-x1)+bow_b
            tip[0] = (x2-x1)
            tip += (x1,y1)
        elif condition2 and instrument != 'cello':
            if handpos == (0,0):
                tip[1] = (y2-y1)
                tip[0] = ((y2-y1)-bow_b)/bow_k
                tip += (x1,y1)
                tip = np.array((min(tip[0],x2),min(tip[1],y2)))
            elif handpos == (0,y2-y1):
                tip[1] = 0 
                tip[0] = (0-bow_b)/bow_k
                tip += (x1,y1)
                tip = np.array((min(tip[0],x2),max(tip[1],y1)))
        elif condition3:
            if handpos == (0,0):
                tip = (x2,y2)
            elif handpos == (0,y2-y1):
                tip = (x2,y1)
            elif handpos == (x2-x1,0):
                tip = (x1,y2)
            elif handpos == (x2-x1,y2-y1):
                tip = (x1,y1)
        else:
            if handpos == (0,0):
                tip = (min(tip[0],x2),min(tip[1],y2))
            elif handpos == (0,y2-y1):
                tip = (min(tip[0],x2),max(tip[1],y1))
            elif handpos == (x2-x1,0):
                tip = (max(tip[0],x1),min(tip[1],y2))
            elif handpos == (x2-x1,y2-y1):
                tip = (max(tip[0],x1),max(tip[1],y1))
    
    wrongflag = (np.array((tip))< 0).any()
    if wrongflag :        
        if handpos == (0,0): 
            tip = np.array(((x2-x1)*0.95,(y2-y1)*0.95))
        elif handpos == (0,y2-y1):
            tip = np.array(((x2-x1)*0.95,0))
        tip += (x1,y1)
    
    #improve frog
    frog += (x1,y1)
    with open(f'../human_2d_result/{proj_dir}/{cam_num}/{video_num + 1}.json','r') as f:
        human2D_data = np.asarray(json.load(f))#,dtype = np.int32
    f.close()

    if instrument == 'cello':
        frog_pos = [
                    (human2D_data[128-1][:2]+human2D_data[124-1][:2])/2
                    ]
    else:
        frog_pos = [(human2D_data[126-1][:2]+human2D_data[122-1][:2])/2,
                    (human2D_data[127-1][:2]+human2D_data[123-1][:2])/2,
                    (human2D_data[128-1][:2]+human2D_data[124-1][:2])/2,
                    (human2D_data[129-1][:2]+human2D_data[125-1][:2])/2
                    ]
    
    if not wrongflag:
        rank_frog_angels = []
        for i in range(len(frog_pos)):
            rank_frog_angels.append(abs(cal_angle(frog,frog_pos[i])-detected_line_angle))
        
        # previous_frog = frog.copy()
        frog_id = np.argmin(rank_frog_angels)
        frog = frog_pos[frog_id]
        '''
        bow_res = np.array([previous_frog,previous_tip]).reshape(1, 2, 2)
        plot_images([image], ['previous lines'], cmaps='gray')
        plot_lines([bow_res], line_colors='red', indices=range(1))
        plt.show()


        bow_res = np.array([frog,tip]).reshape(1, 2, 2)
        plot_images([image], ['improved lines'], cmaps='gray')
        plot_lines([bow_res], line_colors='green', indices=range(1))
        plt.show()
        '''

        return np.asarray(frog),np.asarray(tip),frog_id
    else:
        frog = frog_pos[previous_frog_id]
        '''
        bow_res = np.array([previous_frog,previous_tip]).reshape(1, 2, 2)
        plot_images([image], ['previous lines'], cmaps='gray')
        plot_lines([bow_res], line_colors='red', indices=range(1))
        plt.show()


        bow_res = np.array([frog,tip]).reshape(1, 2, 2)
        plot_images([image], ['improved lines'], cmaps='gray')
        plot_lines([bow_res], line_colors='green', indices=range(1))
        plt.show()
        '''
        return np.asarray(frog),np.asarray(tip),previous_frog_id
##################### Some methods for Processing Keypoints on Bow #####################


def var_to_dict(**kwargs):
    return kwargs



if __name__ == '__main__':
    inform = {}
    
    instrument = 'cello'
    inform.update(var_to_dict(instrument  = instrument))
    
    start_frame_idx = 608
    inform.update(var_to_dict(start_frame_idx = start_frame_idx))

    end_frame_idx = 908
    inform.update(var_to_dict(end_frame_idx = end_frame_idx))
    
    iter_frames = 500 # Number of iteration frames per model insertion <=video.count_frames()
    inform.update(var_to_dict(iter_frames = iter_frames))
    
    proj_dir = 'chuizhenanfeng'
    inform.update(var_to_dict(proj_dir = proj_dir))
    
    video_path =  r'../data/cello_0327/chuizhenanfeng/videos/chuizhenanfeng_21334181.avi'
    inform.update(var_to_dict(video_path = video_path))
    
    cam_num = video_path.split('_')[-1].split('.')[0]
    inform.update(var_to_dict(cam_num = cam_num))

    parent_folder = os.path.dirname(video_path)
    base_name = os.path.basename(video_path)

    labeled_json = f'labeled_jsons/{proj_dir}/{cam_num}/{cam_num}_{start_frame_idx}.json'
    inform.update(var_to_dict(labeled_json = labeled_json))

    inform.update(get_seperate_list(inform))
    
    # Assign graphics card for torch
    if torch.cuda.is_available():
        if torch.cuda.device_count() == 1:
            torch_device = torch.device('cuda')
        else:
            torch_device = torch.device('cuda:{}'.format(torch.cuda.device_count()-1))
        torch_device_ids = np.arange(torch.cuda.device_count())
    else:
        torch_device = torch.device('cpu')
    inform.update(var_to_dict(torch_device = torch_device))
    
    
    # Load the checkpoint (TAP)
    print('Loading the checkpoint...')
    TAPIR_ckpt_path = os.path.abspath(".") + os.sep + 'tapnet/checkpoints/causal_tapir_checkpoint.npy'
    #TAPIR_checkpoint_path = os.path.abspath(".") + os.sep + 'tapnet/checkpoints/bootstapir_checkpoint.npy'
    inform.update(var_to_dict(TAPIR_ckpt_path = TAPIR_ckpt_path))
    TAPIR_model_type = os.path.basename(TAPIR_ckpt_path).split('_')[0]
    inform.update(var_to_dict(TAPIR_model_type = TAPIR_model_type))

    if not os.path.exists(TAPIR_ckpt_path):
        TAPIR_download_checkpoint(inform)

    # Build the model (TAP)
    print('Building the model...')
    if TAPIR_model_type == 'causal':
        TAPIR_model, TAPIR_online_model_init, TAPIR_online_model_predict = TAPIR_test_checkpoint(inform)
        inform.update(var_to_dict(TAPIR_online_model_init = TAPIR_online_model_init))
        inform.update(var_to_dict(TAPIR_online_model_predict = TAPIR_online_model_predict))
    else:
        TAPIR_model, TAPIR_inference = TAPIR_test_checkpoint(inform)
        inform.update(var_to_dict(TAPIR_inference = TAPIR_inference))
    inform.update(var_to_dict(TAPIR_model = TAPIR_model))


    # Track[Infer] (TAP)
    # ------------------------------------------------------------------------
    if instrument == 'cello':

        ROI_size = 512
        inform.update(var_to_dict(ROI_size = ROI_size))

        resize_pixel = 512
        inform.update(var_to_dict(resize_pixel = resize_pixel))


        instrument_kps = ['scroll_top', 'nut_l', 'nut_r']
        inform.update(var_to_dict(instrument_kps = instrument_kps))

        guided_kps = ['nut_guide']#'nut_guide'
        inform.update(var_to_dict(guided_kps = guided_kps))

        inform.update(get_origin(inform))

        inform.update(TAPIR_infer(inform))

        insturment_results = inform['instrument_kp_tracks']+inform['origin']


        # ------------------------------------------------------------------------


        ROI_size = 512
        inform.update(var_to_dict(ROI_size = ROI_size))

        resize_pixel = 512
        inform.update(var_to_dict(resize_pixel = resize_pixel))


        instrument_kps = ['bridge_l', 'bridge_r']
        inform.update(var_to_dict(instrument_kps = instrument_kps))

        guided_kps = ['bridge_guide']#'bridge_guide'
        inform.update(var_to_dict(guided_kps = guided_kps))

        inform.update(get_origin(inform))

        inform.update(TAPIR_infer(inform))

        insturment_results = np.concatenate((insturment_results,inform['instrument_kp_tracks']+inform['origin']), axis=0)

        # ------------------------------------------------------------------------

        ROI_size = 512
        inform.update(var_to_dict(ROI_size = ROI_size))

        resize_pixel = 512
        inform.update(var_to_dict(resize_pixel = resize_pixel))


        instrument_kps = ['tail_gut', 'end_pin']
        inform.update(var_to_dict(instrument_kps = instrument_kps))

        guided_kps = []
        inform.update(var_to_dict(guided_kps = guided_kps))

        inform.update(get_origin(inform))

        inform.update(TAPIR_infer(inform))

        insturment_results = np.concatenate((insturment_results,inform['instrument_kp_tracks']+inform['origin']), axis=0)


    # ------------------------------------------------------------------------
    else:
        iter_frames = 200 # Number of iteration frames per model insertion <=video.count_frames()
        inform.update(var_to_dict(iter_frames = iter_frames))

        inform.update(get_seperate_list(inform))

        ROI_size = 1024
        inform.update(var_to_dict(ROI_size = ROI_size))

        resize_pixel = 1024
        inform.update(var_to_dict(resize_pixel = resize_pixel))


        instrument_kps = ['scroll_top', 'nut_l', 'nut_r']#+['bridge_l', 'bridge_r']
        inform.update(var_to_dict(instrument_kps = instrument_kps))

        guided_kps = ['nut_guide']#'nut_guide'
        inform.update(var_to_dict(guided_kps = guided_kps))

        inform.update(get_origin(inform))

        inform.update(TAPIR_infer(inform))

        insturment_results = inform['instrument_kp_tracks']+inform['origin']


        # ------------------------------------------------------------------------

        iter_frames = 500 # Number of iteration frames per model insertion <=video.count_frames()
        inform.update(var_to_dict(iter_frames = iter_frames))

        inform.update(get_seperate_list(inform))

        ROI_size = 512
        inform.update(var_to_dict(ROI_size = ROI_size))

        resize_pixel = 512
        inform.update(var_to_dict(resize_pixel = resize_pixel))


        instrument_kps = ['bridge_l', 'bridge_r']
        inform.update(var_to_dict(instrument_kps = instrument_kps))

        guided_kps = ['bridge_guide']#'bridge_guide'
        inform.update(var_to_dict(guided_kps = guided_kps))

        inform.update(get_origin(inform))

        inform.update(TAPIR_infer(inform))

        insturment_results = np.concatenate((insturment_results,inform['instrument_kp_tracks']+inform['origin']), axis=0)



    # ------------------------------------------------------------------------
    # That's it!
    
    # YOLOv8 & DeepLSD
    # ------------------------------------------------------------------------
    # YOLOv8 checkpoint loading
    YOLOv8_ckpt_path = os.path.abspath('.')+'/yolov8/checkpoints'
    if not os.path.exists(YOLOv8_ckpt_path):
        if not os.path.exists(os.path.dirname(YOLOv8_ckpt_path)):
            os.makedirs(os.path.dirname(YOLOv8_ckpt_path), exist_ok=True)
    YOLOv8_ckpt = YOLO(YOLOv8_ckpt_path+os.sep+'bow_detection.pt')
    inform.update(var_to_dict(YOLO_conf_threshhold = 0.25))
    
    # DeepLSD checkpoint loading
    DeepLSD_ckpt_path  = os.path.abspath('.')+'/deeplsd/checkpoints/deeplsd_md.tar'
    inform.update(var_to_dict(DeepLSD_ckpt_path =  DeepLSD_ckpt_path))
    DeepLSD_model_type = os.path.splitext(os.path.basename(DeepLSD_ckpt_path))[0].split('_')[-1]
    inform.update(var_to_dict(DeepLSD_model_type =  DeepLSD_model_type))

    if not os.path.exists(DeepLSD_ckpt_path):
        if not os.path.exists(os.path.dirname(DeepLSD_ckpt_path)):
            os.makedirs(os.path.dirname(DeepLSD_ckpt_path), exist_ok=True)
        DeepLSD_download_checkpoint(inform)
    
    video = imageio.get_reader(os.path.abspath(video_path),  'ffmpeg')
    
    DeepLSD_model = DeepLSD_test_checkpoint(inform)
    inform.update(var_to_dict(DeepLSD_model = DeepLSD_model))
    
    previous_conf = 0
    current_conf = 0
    frog_results = None
    tip_results = None
    previous_tip = None
    previous_frog_id = 0
    
    for num in tqdm(range(end_frame_idx), desc="(YOLO & DeepLSD) Loading frames & Inferring"):
        if (num + 1) >= start_frame_idx:

            image = np.asarray(video.get_data(num), dtype=np.uint8)
            image = frame_rotate(cam_num, image)

            #YOLO
            YOLO_results = YOLOv8_ckpt.predict(image.copy()[:,:,::-1],conf = inform['YOLO_conf_threshhold'],
                                       imgsz=640,device = torch_device,verbose = False)
            num_bbox = len(YOLO_results[0].boxes.cls)
            #print('bbox_num',num_bbox)
            if num_bbox >= 1:
                inform.update(var_to_dict(bow_bbox = YOLO_results[0].boxes[0].cpu()))

                current_conf = inform['bow_bbox'].conf.float()[0]#YOLO_results[0].boxes[0].conf.cpu().numpy()[0]
                bow_bbox_xyxy = inform['bow_bbox'].xyxy.numpy().flatten()  #YOLO_results[0].boxes[0].xyxy.cpu().numpy().flatten()

                bow_bbox_xyxy_int32 = bow_bbox_xyxy.astype('uint32')

            else:
                current_conf = inform['bow_bbox'].conf.float()[0] if inform['bow_bbox'].conf.float()[0] >= inform['YOLO_conf_threshhold'] else 0
                bbox_border = 10
                if current_conf == 0 and previous_conf != 0:
                    inform['bow_bbox'].xyxy.numpy().flatten()[:2 ] -= bbox_border
                    inform['bow_bbox'].xyxy.numpy().flatten()[-2:] += bbox_border
                bow_bbox_xyxy = inform['bow_bbox'].xyxy.numpy().flatten()
                bow_bbox_xyxy_int32 = bow_bbox_xyxy.astype('uint32')

            previous_conf = current_conf

            #DeepLSD
            DeepLSD_infer_image = image[bow_bbox_xyxy_int32[1]:bow_bbox_xyxy_int32[3],
                                        bow_bbox_xyxy_int32[0]:bow_bbox_xyxy_int32[2],:]
            #DeepLSD_infer_image = adjust_image_factor(DeepLSD_infer_image, contrast=10, brightness=0.75)

            inform.update(var_to_dict(DeepLSD_infer_image = DeepLSD_infer_image))

            pred_lines = DeepLSD_infer(inform)
            longest_line,pred_lines = compute_longest_line(inform,pred_lines)

            handpos = verify_handpos(inform,longest_line,num)
            frog,tip = detect_frog_tip(inform,image,handpos,longest_line)

            frog,tip,previous_frog_id = improved_frog_tip(inform,num,frog,tip,handpos,image,previous_frog_id)
            bow_result = np.concatenate(([frog],[tip]),axis = 0)[:,np.newaxis,:]
            bow_conf = np.ones((bow_result.shape[0],1))
            #print(frog,tip)
            if (num + 1) == start_frame_idx:
                bow_results = bow_result
                bow_confs = bow_conf
            else:
                bow_results = np.concatenate((bow_results,bow_result),axis = 1)
                bow_confs = np.concatenate((bow_confs,bow_conf),axis = 0)
            #break
    
    # ------------------------------------------------------------------------
    # That's it!
    
    # Final_results
    all_results = np.concatenate((insturment_results,bow_results),axis = 0)
    
    #Visualize
    # ------------------------------------------------------------------------    
    colormap = viz_utils.get_colors(all_results.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_size = tuple(np.flip(video.get_data(0).shape)[1:])  # tuple->(width,height)
    
    out = cv2.VideoWriter(f'chuizhenanfeng_{cam_num}_{TAPIR_model_type}.avi', fourcc, fps=30, frameSize= np.flip(frame_size))# frame.shape[0:2]
    #print(f'{proj_dir}/{file_name}.avi')

    # Visualize and generate a video.
    for num in tqdm(range(end_frame_idx),desc=f'Create the inferring results of video for "Camera:{cam_num}"'):
        if num >= start_frame_idx:
            image = np.asarray(video.get_data(num), dtype=np.uint8)
            image = frame_rotate(cam_num, image)#[origin[1]:origin[1]+ROI_size,origin[0]:origin[0]+ROI_size,:]
            image = cv2.putText(image, str(num-start_frame_idx+1), (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 2)
            for j, color in enumerate(colormap):
                frame = cv2.circle(image,
                                   tuple(np.array(all_results[j][num-start_frame_idx+1],#+origin
                                                  dtype=np.uint32)), 1, color, 10)
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # ------------------------------------------------------------------------
    # That's it!
    
    # Get the outputs
    # ------------------------------------------------------------------------
    visibles_results = np.ones((all_results.shape[:2]))[:, :, np.newaxis]
    pos = np.concatenate((all_results, visibles_results), axis=2).transpose(1, 0, 2)

    save_folder_path = './kp_result'
    save_sub_sub_dir_path = save_folder_path + os.sep + proj_dir + os.sep + cam_num
    if not os.path.exists(save_sub_sub_dir_path):
        os.makedirs(save_sub_sub_dir_path, exist_ok=True)

    for idx, kp_info in enumerate(pos):
        with open(f'{save_folder_path}/{proj_dir}/{cam_num}/{start_frame_idx + idx}.json', 'w') as f:
            f.write(json.dumps(kp_info.tolist()))
        f.close()
    
    # ------------------------------------------------------------------------
    # That's it!
