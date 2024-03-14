import os
"""
The script facilitates a clearer and faster execution of the project.
This is the FIFTH script.
You may need to edit: parent_dir, proj_dir, end_frame_idx, visualize, draw_cps, draw_filtered_cps
5. human and instrument key points should be triangulated.
6. CONTACT POINTS DETECTION
"""


parent_dir = 'cello_1113'
proj_dir = 'cello_1113_pgy'
wave_file = 'wavs/pgy_133_647.wav'
visualize = False  # whether to visualize the 3D representation of smoothed dw outcomes
draw_cps = False  # whether to draw contact points on virtual finger board
draw_filtered_cps = True  # whether to draw filtered contact points on virtual finger board

# parent_dir = 'cello_0111'
# proj_dir = 'aidelizan'
# start_frame_idx = 1211
# end_frame_idx = 1214  # could be a bit bigger than the exact end frame
# cam_file = 'jsons/cello_0111_camera.json'

"""
INFER
2D key points on human are collected in this section.
Multi-view 2D key points will be triangulated to obtain the 3D coordinates.
"""

os.chdir('./audio/')

cp_detection_command = f'python contact_points_pipeline.py ' \
                        f'--wav_path {wave_file} ' \
                        f'--proj_dir {proj_dir}'
if visualize:
    cp_detection_command += ' --visualize'
if draw_cps:
    cp_detection_command += ' --draw_cps'
if draw_filtered_cps:
    cp_detection_command += ' --draw_filtered_cps'

os.system(cp_detection_command)
