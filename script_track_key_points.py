import os
"""
The script facilitates a clearer and faster execution of the project.
This is the THIRD script.
You may need to edit: parent_dir, proj_dir, start_frame_idx, end_frame_idx, instrument, track_cam
3.1. The cello/violin key points should be manually labeled. (Pre-request)
4. INFER human key points (Pre-request)
5. TRACK instrument key points (Run this script)
6. TRIANGULATION (Next script)
"""


parent_dir = 'cello_1113'
proj_dir = 'cello_1113_pgy'
start_frame_idx = 127  # obtained by audio and video alignment
end_frame_idx = 658
instrument = 'cello'
track_cams = ['21334181', '21334190', '21334237']  # cello
# track_cams = ['21334220', '21334207']  # violin


"""
TRACK KEY POINTS
2D key points on cello/violin are collected in this section.
Multi-view 2D key points will be triangulated to obtain the 3D coordinates.
"""

os.chdir('./cello_kp_2d/')
for track_cam in track_cams:
    video_path = f'../data/{parent_dir}/{proj_dir}/videos/{proj_dir}_{track_cam}.avi'

    trackkeypoint_command = f'python TrackKeypoints_pipeline.py ' \
                            f'--proj_dir {proj_dir} ' \
                            f'--video_path {video_path} ' \
                            f'--start_frame_idx {start_frame_idx} ' \
                            f'--end_frame_idx {end_frame_idx} ' \
                            f'--instrument {instrument}'
    os.system(trackkeypoint_command)
