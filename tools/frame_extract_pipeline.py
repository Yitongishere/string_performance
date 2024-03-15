import argparse

import cv2
import os
import re
import glob
from icecream import ic


def getFiles(dir, suffix):  # find the files with specific suffix
    res = []
    for root, directory, files in os.walk(dir):
        for filename in files:
            name, suf = os.path.splitext(filename)
            if suf == suffix:
                res.append(os.path.join(root, filename))
    return res


# extract frames every x frames / seconds
def extract_frames(video_path, output_path, interval, camera_i, bysec=False):
    os.makedirs(output_path, exist_ok=True)
    # open the video file
    video = cv2.VideoCapture(video_path)

    # get the frame rate / sec
    fps = video.get(cv2.CAP_PROP_FPS)
    # print(f"\t Frame rate is: {fps}")

    # calcu frame interval
    if bysec == True:
        frame_interval = int(fps * interval)
    else:
        frame_interval = interval

    # count
    frame_count = 0
    num = 0
    while True:
        # get 1 frame
        ret, frame = video.read()

        # if no frame afterward (end of the video), break
        if not ret:
            break

        # get frame at the beginning of the frame_interval
        if frame_count % frame_interval == 0:
            # save path for extracted frame
            save_path = os.path.join(output_path, f"{camera_i}_{num}.jpg")
            # two cameras with special placement
            if (video_path == r'../data/calib/calib_0/calib_0_21334181.avi' or
                    video_path == r'../data/calib/calib_0/calib_0_21334237.avi'):
                frame_trans = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            else:
                frame_trans = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(save_path, frame_trans)
            num += 1
        # next frame
        frame_count += 1

    video.release()


# extract specific frames listed in the "frame_list"
def extract_spec_frames(video_path, output_path, camera_num, frame_list=[1, 200, 250]):
    os.makedirs(output_path, exist_ok=True)
    # open the video file
    video = cv2.VideoCapture(video_path)

    frame_count = 0

    frame_list_been = []

    while True:
        # get 1 frame
        ret, frame = video.read()

        # if no frame afterward (end of the video), break
        if not ret:
            break

        frame_count += 1
        # get frame at the beginning of the frame_interval
        if frame_count in frame_list:
            frame_list_been.append(frame_count)
            # save path for extracted frame
            save_path = os.path.join(output_path, f"{camera_num}_{frame_count}.jpg")
            # two cameras with special placement
            if re.search('21334181', video_path) or re.search('21334237', video_path):
                frame_trans = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            else:
                frame_trans = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(save_path, frame_trans)
            print(f'{camera_num} frame {frame_count} extracted ...')
        if len(frame_list_been) == len(frame_list):
            break
    video.release()


# TODO: reconstruct the code
def extract_calib_frame(directory_path, output_path):
    # set which calib set to deal with, specify [0-5]
    calib_set_number = 5
    # path of the videos
    directory_path = f"../data/calib_video/calib_{calib_set_number}"
    # video format
    video_format = ".avi"

    # specify the frames
    frame_list_set = [[70, 80, 90, 100, 110, 130, 140, 150, 300, 310],
                      [10, 40, 70, 100, 130, 160, 190, 220, 250, 260],
                      [10, 20, 70, 100, 130, 160, 190, 240, 250, 280],
                      [10, 40, 70, 100, 130, 160, 190, 220, 250, 280],
                      [10, 50, 70, 100, 130, 160, 190, 220, 250, 260],
                      [10, 40, 70, 100, 130, 160, 190, 220, 250, 260]]

    # output path
    output_path = f"../calib_frames"

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    output_path += os.sep + f"calib_frames_{calib_set_number}"

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    frame_list = frame_list_set[calib_set_number]
    for i, file in enumerate(getFiles(directory_path, video_format)):
        save_path = output_path
        cam_num = file.split('_')[-1].split('.')[0]
        print(cam_num)
        save_path = os.path.join(save_path, cam_num)
        print(save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        extract_spec_frames(file, save_path, cam_num, frame_list)
        print(f"Frames extraction in '{file}' finished")

    # set the interval (frame / sec)
    # interval = 1
    # for i, file in enumerate(getFiles(directory_path, video_format)):
    #     extract_frames(file, output_path, interval, i, bysec=True)
    #     print(f"The {i}th camera finished")


# Extract frames interval: [start : end-1]
def extract_frames_interval(files_path, output_path, frame_list):
    videos_path = glob.glob(files_path)
    base_name = [os.path.basename(i) for i in videos_path]
    file_name = [os.path.splitext(i)[0] for i in base_name]
    cam_nums = [i.split('_')[-1] for i in file_name]
    # frame_list = [i for i in range(start, end)]

    print('videos_path:', videos_path)
    print('frame_list:', frame_list)

    for i, video_path in enumerate(videos_path):
        # save_path = f'{output_path}/{cam_num[i]}'
        # if not os.path.exists(save_path):
        #     os.mkdir(save_path)
        frames_output_path = output_path + os.sep + cam_nums[i]
        extract_spec_frames(video_path, frames_output_path, cam_nums[i], frame_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='frame_extract_pipeline')
    parser.add_argument('--files_path', default='../data/cello_1113/cello_1113_scale/video/cello_1113_21334190.avi', type=str, required=True)
    parser.add_argument('--output_path', default='../data/cello_1113/cello_1113_scale/frames', type=str, required=True)
    parser.add_argument('--extract_frames', default='[128]', type=int, nargs='+', required=True)
    # parser.add_argument('--start_frame_idx', default='1', type=int, required=True)
    # parser.add_argument('--end_frame_idx', default='2', type=int, required=True)
    args = parser.parse_args()
    files_path = args.files_path
    output_path = args.output_path
    extract_frames = args.extract_frames
    # start_frame_idx = args.start_frame_idx
    # end_frame_idx = args.end_frame_idx

    # path of videos to be extracted
    # files_path = f"../data/cello_1113/cello_1113_pgy/video/*.avi"
    # files_path = f"../data/cello_1113/cello_1113_scale/video/cello_1113_21334190.avi"

    # output path for extracted frames
    # output_path = f"../data/cello_1113/cello_1113_pgy/frames"
    # output_path = f"../data/cello_1113/cello_1113_scale/frames"

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # print(extract_frames)

    extract_frames_interval(files_path, output_path, extract_frames)
