import os

"""
The script facilitates a clearer and faster execution of the project.
This is the FOURTH script.
You may need to edit: parent_dir, proj_dir, start_frame_idx, end_frame_idx, resolve, xml_path, json_path
3.2. Camera Calibrated. (Pre-request)
4. Human key points should be inferred. (Pre-request)
5. Instrument key points should be tracked. (Pre-request)
6. TRIANGULATION (Run this script)
7. CONTACT POINTS DETECTION (Next script)
"""

parent_dir = 'cello_1113'
proj_dir = 'cello_1113_pgy'
start_frame_idx = 133
end_frame_idx = 647
# end_frame_idx = 134
resolve = False  # whether to resolve xml file
if resolve:
    xml_path = 'xmls/cello_1113_pgy_camera.xml'
json_path = 'jsons/cello_1113_scale_camera.json'

"""
TRIANGULATION
In this section, 3D coordinates of human body, instrument body, and bow are triangulated based on Multi-view 2D key points.
"""

os.chdir('./triangulation/')

if resolve:
    xml_resolve_command = f'python camera_xml_resolve_pipeline.py ' \
                          f'--xml_path {xml_path} ' \
                          f'--json_path {json_path} '
    os.system(xml_resolve_command)
    print(f'Cam file {xml_path} has been resolved as {json_path}.')

triangulation_command = f'python triangulation_pipeline.py ' \
                        f'--cam_file {json_path} ' \
                        f'--parent_dir {parent_dir} ' \
                        f'--proj_dir {proj_dir} ' \
                        f'--start_frame {start_frame_idx} ' \
                        f'--end_frame {end_frame_idx}'
os.system(triangulation_command)
