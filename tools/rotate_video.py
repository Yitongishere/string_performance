import cv2
import os

"""
    给孙老师跑Hand Pose Estimation前的视频筛选和旋转
"""

def rotate_video(input_path, output_path, angle):
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"无法打开视频文件：{input_path}")
        return

    # 获取视频的基本信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建 VideoWriter 对象，用于保存旋转后的视频
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (height, width))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 旋转帧
        rotated_frame = cv2.rotate(frame, angle)

        # 写入输出视频文件
        out.write(rotated_frame)

    # 释放资源
    cap.release()
    out.release()
    print(f"{input_path.split('_')[-1].split('.')[0]} finished")

if __name__ == "__main__":
    CLOCK_WISE = ['21334181', '21334237']
    SELSECTED_CAM = ['21293325', '21293326', '21334180', '21334181', '21334184', '21334186', '21334190',
                     '21334208', '21334209', '21334211', '21334219', '21334221', '21334237', '21334238']
    OMITTED_CAM = ['21293324', '21334183', '21334191', '21334206', '21334207', '21334220']

    # 输入文件夹路径和输出文件夹路径
    proj_dir = 'cello_1113/cello_1113_pgy'
    input_folder = f"../data/{proj_dir}/video"
    output_folder = f"../data/{proj_dir}/{proj_dir.split('/')[-1]}_videos"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # 遍历输入文件夹中相机编号被选择的所有 AVI 文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".avi") and filename.split('_')[-1].split('.')[0] in SELSECTED_CAM:
            cam_num = filename.split('_')[-1].split('.')[0]
            if cam_num in CLOCK_WISE:
                rotation_angle = cv2.ROTATE_90_CLOCKWISE
            else:
                rotation_angle = cv2.ROTATE_90_COUNTERCLOCKWISE
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            rotate_video(input_path, output_path, rotation_angle)

    print("视频旋转完成！")