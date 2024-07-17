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
from tools.Python_in_Shell import getPython3_command
from tools.load_summary import get_folder, get_inform


shell_python_cmd = getPython3_command()

if __name__ == '__main__':
    instrument = 'cello' # cello or violin
    parent_dir = instrument
    root_path = os.path.abspath(f'./data/{parent_dir}')
    folder_names = get_folder(parent_dir,root_path)
    if instrument == 'cello':
        track_cams = ['21334181','21334190','21334211'] # cello
    else:
        track_cams = ['21334207','21334218','21334220','21293324']] # violin
    
    print(folder_names)
    
    os.chdir('./cello_kp_2d/')
    for folder_name in folder_names:
        summary, summary_jsonpath = get_inform(folder_name,root_path)
        proj_dir = folder_name
        start_frame_idx = summary['StartFrame'] # obtained by audio and video alignment
        end_frame_idx = summary['EndFrame']# obtained by audio and video alignment

        
        """
        TRACK KEY POINTS
        2D key points on cello/violin are collected in this section.
        Multi-view 2D key points will be triangulated to obtain the 3D coordinates.
        """

      
        for track_cam in track_cams:
            video_path = os.path.abspath(f'../data/{parent_dir}/{proj_dir}/{proj_dir}_{track_cam}.avi')
            prompt = f'{proj_dir}_{track_cam}.avi'
            print(" "*8 + "-"*(len(prompt)+4)+"\n"
                  f"Target: | {prompt} |\n"+
                  " "*8 + "-"*(len(prompt)+4))
            #TrackKeypoints_pipeline.py
            trackkeypoint_command = f'{shell_python_cmd} TrackKeypoints_pipeline.py ' \
                                    f'--parent_dir {parent_dir} ' \
                                    f'--proj_dir {proj_dir} ' \
                                    f'--video_path {video_path} ' \
                                    f'--start_frame_idx {start_frame_idx} ' \
                                    f'--end_frame_idx {end_frame_idx} ' \
                                    f'--instrument {instrument}'
            os.system(trackkeypoint_command)
