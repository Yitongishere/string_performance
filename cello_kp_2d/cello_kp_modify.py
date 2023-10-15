import glob
import os
import json

'''
Modify the following jsons according to the first cello body key points position.
Please store the cello key points json in directory cello_kp_2d.
Jsons from same camera should be saved in the same directory.
For example: cello_kp_2d/camera_21334181
'''


def edit_json_data(path, cello):
    with open(path, 'rb') as f:
        content = json.load(f)
        for key in cello.keys():
            key_points = {
                "label": key,
                "points": [cello[key]],
                "group_id": None,
                "description": "",
                "shape_type": "point",
                "flags": {}
            }
            # for/else: no encounter break then execute else
            for i, shape in enumerate(content['shapes']):
                if shape['label'] == key:
                    content['shapes'][i] = key_points
                    break
            else:
                content['shapes'].append(key_points)
    return content


def write_json_data(path, data):
    print(f'Write to {path}...')
    with open(path, 'w') as r:
        json.dump(data, r, indent=4)


if __name__ == '__main__':
    dirs = []
    for item in os.scandir('.'):
        if item.is_dir():
            dirs.append(item.path)
    for dir_path in dirs:
        files = glob.glob(os.path.join(dir_path, '*.json'))
        files = sorted(files, key=lambda name: int(name.split('_')[-1].split('.')[0]))
        cello_dict = {
            'scroll_top': None,
            'nut_l': None,
            'nut_r': None,
            'neck_bottom_l': None,
            'neck_bottom_r': None,
            'bridge_l': None,
            'bridge_r': None,
            'tail_gut': None,
            'end_pin': None
        }
        # obtain cello body data from the first file
        labelme = json.load(open(files[0]))
        for each_ann in labelme['shapes']:
            if each_ann['shape_type'] == 'point':
                kpt_label = each_ann['label']
                if kpt_label != 'frog' and kpt_label != 'tip_plate':
                    if cello_dict[kpt_label] is None:
                        kpt_xy = each_ann['points'][0]
                        cello_dict[kpt_label] = [kpt_xy[0], kpt_xy[1]]
        if None not in cello_dict.values():
            # skip the first file
            for file in files[1:]:
                edited_data = edit_json_data(file, cello_dict)
                write_json_data(file, edited_data)
