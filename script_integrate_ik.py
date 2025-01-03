"""
The script facilitates a clearer and faster execution of the project.
This is the SIXTH script.
You may need to edit: parent_dir, proj_dir, cam_file, start_frame_idx, end_frame_idx, visualize, cam_num, instrument
7. CONTACT POINTS should be detected. (Pre-request)
8. INTEGRATE the hand poses by HPE. (Run this script)[The 1st step in this script]
9. INVERSE KINEMATIC. (Run this script)[The 2nd step in this script]
"""

import os
from tools.Python_in_Shell import getPython3_command
from tools.load_summary import get_folder, get_inform, get_folder_extra
from multiprocessing import Pool


def integrate_ik_process(folder_name):
    summary, summary_jsonfile_path = get_inform(folder_name, root_path)
    proj_dir = folder_name
    
    start_frame_idx = summary['StartFrame']
    end_frame_idx = summary['EndFrame']
    cam_parm = summary['CameraParameter']
    
    print(f'Processing FolderName:{summary["FolderName"]}')
    print(f'Track:{summary["Track"]}')
    print(f'start_frame_idx:{start_frame_idx}')
    print(f'end_frame_idx:{end_frame_idx}')
    
    if instrument == 'cello':
        cam_num = 'cam0'
    else:
        cam_num = 'cam10'
    
    # If you want to load the bone length of smplh/smplx: please set [use_defined_bone_length] as 0
    use_defined_bone_length = 1
    # Please set the appendix to the mano file. The files are roughly named f"J3_left_{TYPE}" and "J3_right_{TYPE}" ,The TYPE can be 'CUSTOMED', 'SMPLX',  Or other custom types.
    if use_defined_bone_length:
        mano_file_appendix = 'CUSTOMED'
    else:
        mano_file_appendix = 'SMPLX'
    
    print(f'use_defined_bone_length[ik]:{bool(use_defined_bone_length)}')
    print(f'mano_file_appendix[fk,ik]:[{mano_file_appendix}]')
    
    """
    INTEGRATE handpose
    6D hand poses are regressed by HPE.
    Further, handposes are integrated from different views in this section.
    """
    
    integrate_command = f'{shell_python_cmd} integrate_handpose_pipeline.py ' \
                        f'--summary_jsonfile {summary_jsonfile_path} ' \
                        f'--parent_dir {parent_dir} ' \
                        f'--proj_dir {proj_dir} ' \
                        f'--start_frame {start_frame_idx} ' \
                        f'--end_frame {end_frame_idx} ' \
                        f'--instrument {instrument} ' \
                        f'--mano_file_appendix {mano_file_appendix} ' \
                        f'--cam_num {cam_num}'
                        #f'--end_frame {end_frame_idx}'
    
    os.system(integrate_command)

    """
    INVERSE KINEMATIC
    Integrated handposes are iked based on contact points.
    """
    
    ik_command =    f'{shell_python_cmd} inverse_kinematic_pipeline.py ' \
                    f'--summary_jsonfile {summary_jsonfile_path} ' \
                    f'--parent_dir {parent_dir} ' \
                    f'--proj_dir {proj_dir} ' \
                    f'--instrument {instrument} ' \
                    f'--start_frame {start_frame_idx} ' \
                    f'--mano_file_appendix {mano_file_appendix} ' \
                    f'--use_defined_bone_length {use_defined_bone_length}'
                    #f'--end_frame {end_frame_idx}'

    os.system(ik_command)
    


shell_python_cmd = getPython3_command()


if __name__ == '__main__':
    instrument = 'cello'
    parent_dir = instrument
    root_path = os.path.abspath(f'./data/{parent_dir}')
    folder_names_parent = get_folder(parent_dir, root_path)
    folder_names_6d = get_folder_extra(f'./pose_estimation/6d_result/{parent_dir}')
    folder_names = sorted(list(set(folder_names_parent) & set(folder_names_6d)))
    print(folder_names)
    
    os.chdir('./pose_estimation/')
    with Pool(processes=os.cpu_count()) as pool: 
        pool.map(integrate_ik_process, folder_names[:-1])
        #pool.map(integrate_ik_process, [folder_names[0]])
