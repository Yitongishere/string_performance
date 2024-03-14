import os
"""
The script facilitates a clearer and faster execution of the project.
This is the THIRD script.
You may need to edit: parent_dir, proj_dir, end_frame_idx
4.2. INFER human key points
5. TRIANGULATION
"""


parent_dir = 'cello_1113'
proj_dir = 'cello_1113_pgy'
# always start from first frame
end_frame_idx = 660  # could be a bit bigger than the exact end frame

# parent_dir = 'cello_0111'
# proj_dir = 'aidelizan_removed'
# # always start from first frame
# end_frame_idx = 1700  # could be a bit bigger than the exact end frame


"""
INFER
2D key points on human are collected in this section.
Multi-view 2D key points will be triangulated to obtain the 3D coordinates.
"""

os.chdir('./human_kp_2d/')
dirs_path = fr'../data/{parent_dir}/{proj_dir}'

infer_command = f'python infer_pipeline.py ' \
                        f'--dirs_path {dirs_path} ' \
                        f'--proj_dir {proj_dir} '\
                        f'--end_frame_idx {end_frame_idx}'
os.system(infer_command)
