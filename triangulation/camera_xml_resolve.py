from xml.dom.minidom import parse
from collections import defaultdict
import re
import numpy as np
import json


def camera_sort(label):
    if re.search('21334181', label):
        return 0
    elif re.search('21334237', label):
        return 1
    elif re.search('21334180', label):
        return 2
    elif re.search('21334209', label):
        return 3
    elif re.search('21334208', label):
        return 4
    elif re.search('21334186', label):
        return 5
    elif re.search('21293326', label):
        return 6
    elif re.search('21293325', label):
        return 7
    elif re.search('21293324', label):
        return 8
    elif re.search('21334206', label):
        return 9
    elif re.search('21334220', label):
        return 10
    elif re.search('21334183', label):
        return 11
    elif re.search('21334207', label):
        return 12
    elif re.search('21334191', label):
        return 13
    elif re.search('21334184', label):
        return 14
    elif re.search('21334238', label):
        return 15
    elif re.search('21334221', label):
        return 16
    elif re.search('21334219', label):
        return 17
    elif re.search('21334190', label):
        return 18
    elif re.search('21334211', label):
        return 19
    elif re.search('21334218', label):
        return 20
    elif re.search('21334182', label):
        return 21
    elif re.search('21334236', label):
        return 22
    elif re.search('21334210', label):
        return 23


def getIntrinsics(sensors):
    intrinsics = defaultdict(list)

    for sensor in sensors:
        resolution = sensor.getElementsByTagName('resolution')
        width = resolution[0].getAttribute('width')
        height = resolution[0].getAttribute('height')
        intrinsics['width'].append(float(width))
        intrinsics['height'].append(float(height))

        f = float(sensor.getElementsByTagName('f')[0].childNodes[0].nodeValue)
        intrinsics['f'].append(f)
        try:
            cx = sensor.getElementsByTagName('cx')[0].childNodes[0].nodeValue
            cx = float(width) / 2 + float(cx)
        except IndexError as e:
            cx = 0
            intrinsics['cx'].append(cx)
        try:
            cy = sensor.getElementsByTagName('cy')[0].childNodes[0].nodeValue
            cy = float(height) / 2 + float(cy)
        except IndexError as e:
            cy = 0
            intrinsics['cy'].append(cy)

        K = [f, 0, cx, 0, f, cy, 0, 0, 1]
        intrinsics['K'].append(K)

        # 径向畸变参数
        try:
            k1 = float(sensor.getElementsByTagName('k1')[0].childNodes[0].nodeValue)
        except IndexError as e:
            k1 = 0
        try:
            k2 = float(sensor.getElementsByTagName('k2')[0].childNodes[0].nodeValue)
        except IndexError as e:
            k2 = 0
        try:
            k3 = float(sensor.getElementsByTagName('k3')[0].childNodes[0].nodeValue)
        except IndexError as e:
            k3 = 0

        #     radial_distortion = [k1, k2, k3]
        #     intrinsics['radiDistort'].append(radial_distortion)

        # 切向畸变参数
        try:
            p1 = float(sensor.getElementsByTagName('p1')[0].childNodes[0].nodeValue)
        except IndexError as e:
            p1 = 0
        try:
            p2 = float(sensor.getElementsByTagName('p2')[0].childNodes[0].nodeValue)
        except IndexError as e:
            p2 = 0
        #     circumferential_distortion = [p1, p2]
        #     intrinsics['circumDistort'].append(circumferential_distortion)

        distort = [k1, k2, p1, p2, k3]
        intrinsics['distort'].append(distort)
    return intrinsics


if __name__ == '__main__':
    input_xml_path = 'xmls/cello_1113_scale_camera.xml'
    output_json_path = "jsons/cello_1113_scale_camera.json"


    dom = parse(input_xml_path)
    elem = dom.documentElement
    sensors = elem.getElementsByTagName('calibration')

    intrinsics = getIntrinsics(sensors)

    cameras = elem.getElementsByTagName('camera')

    out = dict()
    for i in range(20):
        out['cam' + str(i)] = dict()

    for camera in cameras:
        parameters = dict()

        # 属于哪一组的摄像机
        sensor_id = camera.getAttribute('sensor_id')
        sensor_id = int(sensor_id)
        sensor_id = 0

        # print(sensor_id)
        parameters['K'] = intrinsics['K'][sensor_id]

        label = camera.getAttribute('label')
        cam_num = camera_sort(label)

        transform = camera.getElementsByTagName('transform')[0].childNodes[0].nodeValue
        transform = transform.split(' ')
        transform = [float(i) for i in transform]

        transform = np.array(transform)
        transform = transform.reshape(4, 4)
        # 外参数矩阵为transfrom的逆矩阵
        # print(transform)
        extrinsics = np.linalg.inv(transform)
        # extrinsics = transform

        # ic(transform@extrinsics)

        R = extrinsics[0:3, 0:3].reshape(-1).tolist()
        T = extrinsics[0:3, -1].tolist()

        parameters['R'] = R
        parameters['T'] = T
        parameters['imgSize'] = [intrinsics['width'][sensor_id], intrinsics['height'][sensor_id]]
        parameters['distort'] = intrinsics['distort'][sensor_id]

        out['cam' + str(cam_num)] = parameters

        with open(output_json_path, "w", encoding='utf-8') as f:
            json.dump(out, f, indent=4)




