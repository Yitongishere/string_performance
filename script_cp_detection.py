"""
The script facilitates a clearer and faster execution of the project.
This is the FIFTH script.
You may need to edit: parent_dir, proj_dir, wave_file, visualize, draw_cps, draw_filtered_cps, save_position
6. Human and instrument key points should be triangulated. (Pre-request)
7. CONTACT POINTS DETECTION. (Run this script)
8. INTEGRATE hand poses. (Next script)
"""

import os
from tools.Python_in_Shell import getPython3_command
from tools.load_summary import get_folder, get_inform
from multiprocessing import Pool


def cp_detection_process(folder_name):
    #summary, summary_jsonfile_path = get_inform(folder_name, root_path)
    proj_dir = folder_name
    wave_file = f'{root_path}/{proj_dir}/{proj_dir}.wav'
    #start_frame_idx = summary['StartFrame']
    #end_frame_idx = summary['EndFrame']
    #cam_parm = summary['CameraParameter']
    
    #print(f'Processing FolderName:{summary["FolderName"]}')
    #print(f'Track:{summary["Track"]}')
    #print(f'start_frame_idx:{start_frame_idx}')
    #print(f'end_frame_idx:{end_frame_idx}')
    os.chdir('./audio/')
    """
    INFER
    2D key points on human are collected in this section.
    Multi-view 2D key points will be triangulated to obtain the 3D coordinates.
    """
    cp_detection_command =  f'{shell_python_cmd} contact_points_pipeline.py ' \
                            f'--wav_path {wave_file} ' \
                            f'--parent_dir {parent_dir} ' \
                            f'--proj_dir {proj_dir} ' \
                            f'--crepe_backend {crepe_backend}'
    if visualize:
        cp_detection_command += ' --visualize'
    if draw_cps:
        cp_detection_command += ' --draw_cps'
    if draw_filtered_cps:
        cp_detection_command += ' --draw_filtered_cps'
    if save_position:
        cp_detection_command += ' --save_position'
    os.system(cp_detection_command)


shell_python_cmd = getPython3_command()

if __name__ == '__main__':
    instrument = 'cello'
    parent_dir = instrument
    root_path = os.path.abspath(f'./data/{parent_dir}')

    folder_names = get_folder(parent_dir, root_path)

    visualize = False  # whether to visualize the 3D representation of smoothed dw outcomes
    draw_cps = False  # whether to draw contact points on virtual finger board
    draw_filtered_cps = False  # whether to draw filtered contact points on virtual finger board
    save_position = True  # whether to save filtered position

    crepe_backend = 'torch' # 'torch' or 'tensorflow'
    
    with Pool(processes=os.cpu_count()) as pool: 
        pool.map(cp_detection_process, folder_names)
