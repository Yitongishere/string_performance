import os
"""
The script facilitates a clearer and faster execution of the project.
This is the SECOND script.
You may need to edit: parent_dir, proj_dir, end_frame_idx
4. INFER human key points (Run this script)
5. TRACK instrument key points (Next script)
"""


parent_dir = 'violin_0323'
proj_dir = 'gxianshangdeyongtandiao'
# always start from first frame
end_frame_idx = 4900  # could be a bit bigger than the exact end frame

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
                        f'--parent_dir {parent_dir} '\
                        f'--proj_dir {proj_dir} '\
                        f'--end_frame_idx {end_frame_idx}'
os.system(infer_command)
