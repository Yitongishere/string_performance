"""
The script facilitates a clearer and faster execution of the project.
This is the SECOND script.
You may need to edit: parent_dir, proj_dir, end_frame_idx
4. INFER human key points (Run this script)
5. TRACK instrument key points (Next script)
"""

import os
from tools.load_summary import get_folder, get_inform

instrument = 'cello' # cello or violin
folder_names,root_path = get_folder(instrument)
parent_dir = instrument
os.chdir('./human_kp_2d/')
for folder_name in folder_names:
    summary = get_inform(folder_name,root_path)
    proj_dir = folder_name
    # always start from first frame
    start_frame_idx = summary['StartFrame']
    end_frame_idx =  summary['EndFrame']# could be a bit bigger than the exact end frame
    
    '''
    parent_dir = 'cello_1113'
    proj_dir = 'cello_1113_pgy'
    # always start from first frame
    end_frame_idx = 660  # could be a bit bigger than the exact end frame
    '''
    
    # parent_dir = 'cello_0111'
    # proj_dir = 'aidelizan_removed'
    # # always start from first frame
    # end_frame_idx = 1700  # could be a bit bigger than the exact end frame
    
    
    """
    INFER
    2D key points on human are collected in this section.
    Multi-view 2D key points will be triangulated to obtain the 3D coordinates.
    """
    
    
    dirs_path = os.path.abspath(f'../data/{parent_dir}/{proj_dir}/')
    
    infer_command = f'python3 infer_pipeline.py ' \
                            f'--dirs_path {dirs_path} ' \
                            f'--proj_dir {proj_dir} '\
                            f'--end_frame_idx {end_frame_idx} '\
                            f'--start_frame_idx {start_frame_idx}'
    os.system(infer_command)
    #break
