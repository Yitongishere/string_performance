"""
The script facilitates a clearer and faster execution of the project.
This is the FOURTH script.
You may need to edit: parent_dir, proj_dir, start_frame_idx, end_frame_idx, resolve, xml_path, json_path
3.2. Camera Calibrated. (Pre-request)
4. Human key points should be inferred. (Pre-request)
5. Instrument key points should be tracked. (Pre-request)
6. TRIANGULATION (Run this script)
7. CONTACT POINTS DETECTION [python 'script_cp_detection.py'](Next script)
"""

import os
from tools.Python_in_Shell import getPython3_command
from tools.load_summary import get_folder, get_inform
from multiprocessing import Pool


def triangulation_process(folder_name):
    summary, summary_jsonfile_path = get_inform(folder_name, root_path)
    proj_dir = folder_name
    
    start_frame_idx = summary['StartFrame']
    end_frame_idx = summary['EndFrame']
    cam_parm = summary['CameraParameter']
    
    print(f'Processing FolderName:{summary["FolderName"]}')
    print(f'Track:{summary["Track"]}')
    print(f'start_frame_idx:{start_frame_idx}')
    print(f'end_frame_idx:{end_frame_idx}')
    
    
    triangulation_command = f'{shell_python_cmd} triangulation_pipeline.py ' \
                            f'--summary_jsonfile {summary_jsonfile_path} ' \
                            f'--parent_dir {parent_dir} ' \
                            f'--proj_dir {proj_dir} ' \
                            f'--start_frame {start_frame_idx} ' \
                            f'--end_frame {end_frame_idx} ' \
                            f'--instrument {instrument}'
    os.system(triangulation_command)


shell_python_cmd = getPython3_command()

if __name__ == '__main__':
    instrument = 'cello'
    parent_dir = instrument
    root_path = os.path.abspath(f'./data/{parent_dir}')
    
    # if you want to use customized $parent_dir
    # You can refer to the following code:
    # from tools.load_summary import get_folder_extra
    # parent_dir = $parent_dir
    # folder_names = get_folder_extra(parent_dir,root_path)
    folder_names = get_folder(parent_dir, root_path)
    
    print(folder_names)
    
    os.chdir('./triangulation/')
    with Pool(processes=os.cpu_count()) as pool: 
        pool.map(triangulation_process, folder_names)
