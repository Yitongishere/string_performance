import os

from icecream import ic

current_path = os.path.dirname(os.path.abspath(__file__))
"""
The script facilitates a clearer and faster execution of the project.
This is the FIRST script.
You may need to edit: parent_dir, proj_dir, start_frame_idx, end_frame_idx, extract_cams, extract_frames
1. When new data is recorded, the audio and video first need to be aligned. (MANUALLY)
2. FRAME EXTRACT (Run this script)
3.1. Label the cello/violin key points. (MANUALLY)
3.2. Camera Calibration (MANUALLY)
"""

parent_dir = 'cello_0327'
proj_dir = 'chuizhenanfeng_jyt'
start_frame_idx = 607  # obtained by audio and video alignment
end_frame_idx = 3516
step = 100
extract_cams = ['21334181', '21334180', '21334237', '21334190']  # cello
# extract_cams = ['21334220', '21334207', '21334236', '21334218']  # violin
# extract_cams = ['21334190', '21334237']  # cello overlay
# extract_cams = ['21334220']  # violin overlay
# extract_frames = [start_frame_idx]
# extract_frames = [start_frame_idx, 300]  # extract more frames for TRACKKEYPOINTS (more labels)

# extract_frames = [i for i in range(start_frame_idx, end_frame_idx+1, step)]  # for overlay

# post processing
# extract_frames_str = ''
# for frame in extract_frames:
#     extract_frames_str += str(frame)
#     extract_frames_str += ' '
# ic(extract_frames)

"""
FRAME EXTRACT
cello cameras: 21334181, 21334190, 21334237(optional, could be out of focus)
violin cameras: 21334220, 21334207
Extracted frames are used to track key points on the instrument.
"""

for extract_cam in extract_cams:
    files_path = f'{current_path}/data/{parent_dir}/{proj_dir}/videos/{proj_dir}_{extract_cam}.avi'
    output_path = f'{current_path}/data/{parent_dir}/{proj_dir}/frames'

    frame_extract_command = f'python ./tools/frame_extract_pipeline.py ' \
                            f'--files_path {files_path} ' \
                            f'--output_path {output_path} ' \
                            f'--start_frame {start_frame_idx} ' \
                            f'--end_frame {end_frame_idx} ' \
                            f'--step {step}'
    print(frame_extract_command)

    os.system(frame_extract_command)
