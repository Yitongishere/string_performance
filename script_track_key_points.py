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
root_path = os.path.abspath(f'./data/{instrument}')

if instrument == 'cello':
    track_cams = ['21334181', '21334190']  # cello , '21334237','21334206'
else:
    track_cams = ['21334220', '21334207']  # violin

folder_names = get_folder(instrument,root_path)

#If you want to process these data in batches, you can use the following annotated code.
'''

index = -1
batch_size = 24

if (index == -1 and step == 1):
    folder_names = [folder_names[-1]]
elif index == -1:
    folder_names = folder_names[index-batch_size+1:index] + [folder_names[-1]]
else:
    folder_names = folder_names[index-batch_size+1:index+1]
'''

print(folder_names)
parent_dir = instrument
os.chdir('./cello_kp_2d/')
for folder_name in folder_names:
    summary, summary_jsonpath = get_inform(folder_name,root_path)
    proj_dir = folder_name
    start_frame_idx = summary['StartFrame'] # obtained by audio and video alignment
    end_frame_idx =  summary['EndFrame']# obtained by audio and video alignment
    

    """
    parent_dir = 'cello_1113'
    proj_dir = 'cello_1113_pgy'
    start_frame_idx = 127  # obtained by audio and video alignment
    end_frame_idx = 658
    instrument = 'cello'
    track_cams = ['21334181', '21334190', '21334237']  # cello
    # track_cams = ['21334220', '21334207']  # violin
    """
    
    """
    TRACK KEY POINTS
    2D key points on cello/violin are collected in this section.
    Multi-view 2D key points will be triangulated to obtain the 3D coordinates.
    """

  
    for track_cam in track_cams:
        video_path = os.path.abspath(f'../data/{parent_dir}/{proj_dir}/{proj_dir}_{track_cam}.avi')
        #TrackKeypoints_pipeline.py
        trackkeypoint_command = f'python3 TrackKeypoints_pipeline.py ' \
                                f'--proj_dir {proj_dir} ' \
                                f'--video_path {video_path} ' \
                                f'--start_frame_idx {start_frame_idx} ' \
                                f'--end_frame_idx {end_frame_idx} ' \
                                f'--instrument {instrument}'
        os.system(trackkeypoint_command)
    #break
