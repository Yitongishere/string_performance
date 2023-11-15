import cv2
import os

CLOCK_WISE = ['21334181', '21334237']


def video_rotate(cam_num, input_video, output_video):
    if cam_num in CLOCK_WISE:
        os.system(
            'ffmpeg -i {} -vcodec h264 -b:v 0 -c:a copy -vf "transpose=1" {}_rot90.mp4'.format(
                os.path.abspath(input_video),
                os.path.abspath(output_video)))
    else:
        os.system(
            'ffmpeg -i {} -vcodec h264 -b:v 0 -c:a copy -vf "transpose=2" {}_rot270.mp4'.format(
                os.path.abspath(input_video),
                os.path.abspath(
                    output_video)))


def frame_rotate(cam_num, input_frame):
    if cam_num in CLOCK_WISE:
        output_frame = cv2.rotate(input_frame, cv2.ROTATE_90_CLOCKWISE)
    else:
        output_frame = cv2.rotate(input_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return output_frame
