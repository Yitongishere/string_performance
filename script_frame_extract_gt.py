import os

import numpy as np
from icecream import ic

current_path = os.path.dirname(os.path.abspath(__file__))
"""
The script facilitates a clearer and faster execution of the project.
This is the FIRST script.
You may need to edit: parent_dir, proj_dir, start_frame_idx, extract_cams
1. When new data is recorded, the audio and video first need to be aligned. (MANUALLY)
2. FRAME EXTRACT (Run this script)
3. Label the cello/violin key points. (MANUALLY)
"""

parent_dir = 'cello_1113'
proj_dir = 'cello_1113_pgy'
gt_dir = f'gt_cellos_{proj_dir}'
start_frame_idx = 128  # obtained by audio and video alignment
end_frame_idx = 647
extract_cams = ['21334181', '21334180', '21334208', '21334237', '21334209']  # cello
# extract_cams = ['21334208']
# extract_cams = ['21293324', '21334207', '21334210', '21334218', '21334220']  # violin

available_frames = []
position_path = f'./audio/positions_{proj_dir}'
position_file = open(position_path)
positions = position_file.readlines()
for idx, position in enumerate(positions):
    position = position.strip()
    position = position.split(' ')
    for ratio in position:
        if float(ratio) != -1.0:
            available_frames.append(idx + start_frame_idx)
            continue
position_file.close()
print('available_frames:', available_frames)

extract_frames = []
for i in range(start_frame_idx, end_frame_idx+1, 8):
    if i not in available_frames:
        continue
    extract_frames.append(i)
print('extract_frames:', extract_frames)
# extract_frames = [241]  # extract more frames (more labels)

"""
FRAME EXTRACT
cello cameras: 21334181, 21334190, 21334237(optional, could be out of focus)
violin cameras: 21334220, 21334207
Extracted frames are used to track key points on the instrument.
"""

# for extract_cam in extract_cams:
#     for extract_frame in extract_frames:
#         files_path = f'{current_path}/data/{parent_dir}/{proj_dir}/videos/{parent_dir}_{extract_cam}.avi'
#         # output_path = f'{current_path}/data/{parent_dir}/{proj_dir}/frames'
#         # print(files_path)
#         output_path = f'{current_path}/human_kp_2d/kp_result/{gt_dir}'
#
#         frame_extract_gt_command = f'python ./tools/frame_extract_pipeline.py ' \
#                                    f'--files_path {files_path} ' \
#                                    f'--output_path {output_path} ' \
#                                    f'--start_frame_idx {extract_frame} ' \
#                                    f'--end_frame_idx {extract_frame + 1}'
#         os.system(frame_extract_gt_command)
