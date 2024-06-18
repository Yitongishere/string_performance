"""
The script facilitates a clearer and faster execution of the project.
This is the SECOND script.
You may need to edit: parent_dir, proj_dir, end_frame_idx
4. INFER human key points (Run this script)
5. TRACK instrument key points (Next script)
"""

import os
from tools.load_summary import get_folder, get_inform

cuda = 0
writevideo = 0

instrument = 'cello' # cello or violin
root_path = os.path.abspath(f'./data/{instrument}')
folder_names = get_folder(instrument,root_path)

#If you want to process these data in batches, you can use the following annotated code.


index = -1
batch_size = 85

if (index == -1 and batch_size == 1):
    folder_names = [folder_names[-1]]
elif index == -1:
    folder_names = folder_names[index-batch_size+1:index] + [folder_names[-1]]
else:
    folder_names = folder_names[index-batch_size+1:index+1]


print(folder_names)
parent_dir = instrument
os.chdir('./human_kp_2d/')    
for folder_name in folder_names:
    summary, _ = get_inform(folder_name,root_path)
    proj_dir = folder_name
    
    start_frame_idx = summary['StartFrame'] # use annotated data in json file or could start from first frame
    end_frame_idx = summary['EndFrame'] # use annotated data in json file or could be a bit bigger than the exact end frame
    
    
    """
    INFER
    2D key points on human are collected in this section.
    Multi-view 2D key points will be triangulated to obtain the 3D coordinates.
    """
    
    
    dirs_path = os.path.abspath(f'../data/{parent_dir}/{proj_dir}/')
    
    print(f'FolderName:{summary["FolderName"]}')
    print(f'Track:{summary["Track"]}')
    print(f'start_frame_idx:{start_frame_idx}')
    print(f'end_frame_idx:{end_frame_idx}')
    
    infer_command = f'python3 infer_pipeline.py ' \
                            f'--dirs_path {dirs_path} ' \
                            f'--proj_dir {proj_dir} '\
                            f'--end_frame_idx {end_frame_idx} '\
                            f'--start_frame_idx {start_frame_idx} '\
                            f'--cuda {cuda} '\
                            f'--writevideo {writevideo}'
    os.system(infer_command)
    #break
