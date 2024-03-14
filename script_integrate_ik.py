import os
"""
The script facilitates a clearer and faster execution of the project.
This is the SIXTH script.
You may need to edit: parent_dir, proj_dir, end_frame_idx, visualize, draw_cps, draw_filtered_cps
6. CONTACT POINTS should be detected
7. INTEGRATE the hand pose by HPE
8. INVERSE KINEMATIC
"""


parent_dir = 'cello_1113'
proj_dir = 'cello_1113_pgy'
cam_file = '../triangulation/jsons/cello_1113_scale_camera.json'
start_frame_idx = 133
visualize = False  # whether to overlay the 3D result
cam_num = 'cam0'  # choose the camera as the overlay camera
instrument = 'cello'

"""
INTEGRATE handpose
6D hand poses are regressed by HPE.
Further, handposes are integrated from different views in this section.
"""

os.chdir('./pose_estimation/')

integrate_command = f'python integrate_handpose_pipeline.py ' \
                        f'--cam_file {cam_file} ' \
                        f'--parent_dir {parent_dir} ' \
                        f'--proj_dir {proj_dir} ' \
                        f'--start_frame {start_frame_idx} ' \
                        f'--instrument {instrument} ' \
                        f'--cam_num {cam_num}'
os.system(integrate_command)

"""
INVERSE KINEMATIC
Integrated handposes are iked based on contact points.
"""

ik_command = f'python inverse_kinematic_pipeline.py ' \
                        f'--cam_file {cam_file} ' \
                        f'--parent_dir {parent_dir} ' \
                        f'--proj_dir {proj_dir} ' \
                        f'--start_frame {start_frame_idx}'

if visualize:
    ik_command += f' --visualize --cam_num {cam_num}'

os.system(ik_command)
