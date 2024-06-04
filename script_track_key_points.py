"""
The script facilitates a clearer and faster execution of the project.
This is the THIRD script.
You may need to edit: parent_dir, proj_dir, start_frame_idx, end_frame_idx, instrument, track_cam
3.1. The cello/violin key points should be manually labeled. (Pre-request)
4. INFER human key points (Pre-request)
5. TRACK instrument key points (Run this script)
6. TRIANGULATION (Next script)
"""

import os
from tools.load_summary import get_folder, get_inform


instrument = 'cello' # cello or violin

if instrument == 'cello':
    track_cams = ['21334181', '21334190']  # cello , '21334237'
else:
    track_cams = ['21334220', '21334207']  # violin

folder_names,root_path = get_folder(instrument)
summary = get_inform(folder_names[0],root_path)

parent_dir = instrument

for folder_name in folder_names:
    proj_dir = folder_name
    start_frame_idx = summary['StartFrame'] # obtained by audio and video alignment
    end_frame_idx =  summary['EndFrame']# obtained by audio and video alignment

"""
TRACK KEY POINTS
2D key points on cello/violin are collected in this section.
Multi-view 2D key points will be triangulated to obtain the 3D coordinates.
"""

os.chdir('./cello_kp_2d/')
for track_cam in track_cams:
    video_path = f'../data/{parent_dir}/{proj_dir}/{proj_dir}_{track_cam}.avi'
    #TrackKeypoints_pipeline.py
    trackkeypoint_command = f'python3 TrackKeypoints_pipeline.py ' \
                            f'--proj_dir {proj_dir} ' \
                            f'--video_path {video_path} ' \
                            f'--start_frame_idx {start_frame_idx} ' \
                            f'--end_frame_idx {end_frame_idx} ' \
                            f'--instrument {instrument}'
    os.system(trackkeypoint_command)
