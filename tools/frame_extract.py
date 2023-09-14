import cv2
import os

def getFiles(dir, suffix): # 查找根目录，文件后缀 
    res = []
    for root, directory, files in os.walk(dir):  # =>当前根,根下目录,目录下的文件
        for filename in files:
            name, suf = os.path.splitext(filename) # =>文件名,文件后缀
            if suf == suffix:
                res.append(os.path.join(root, filename)) # =>吧一串字符串组合成路径
    return res
 
def extract_frames(video_path, output_path, frame_rate, camera_i):
    # 创建输出路径
    os.makedirs(output_path, exist_ok=True)
 
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
 
    # 获取视频的帧率
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"获取视频的帧率:{fps}")
 
    # 计算每隔多少帧抽取一帧
    frame_interval = int(fps*frame_rate)
    
    # 初始化帧计数器
    frame_count = 0
    num = 0
    while True:
        # 读取视频的一帧
        ret, frame = video.read()
 
        # 如果无法读取到帧，则退出循环
        if not ret:
            break
 
        # 如果帧计数器是frame_interval的倍数，则保存该帧
        if frame_count % frame_interval == 0:
            # 构造保存路径
            save_path = os.path.join(output_path, f"camera_{camera_i}.jpg")
 
            # 保存帧为图片
            cv2.imwrite(save_path, frame)
 
            num = frame_count/82  #视频fps为25，设定为3.3s抽帧，相乘后82.5取的82，具体看自身情况
            print(f"当前处理到第{num}张")
 
        # 增加帧计数器
        frame_count += 1
 
    # 释放视频对象
    video.release()


directory_path = "./dwpose/"    #修改此处为视频存放目录
video_format = ".avi"       #修改此处为视频格式
output_path = "./frames/"  #修改此处为输出路径
frame_rate = 25   #每多少秒抽帧，此处设定为25秒抽取一张图片
for i, file in enumerate(getFiles(directory_path, video_format)):  # =>查找以.py结尾的文件
    extract_frames(file, output_path, frame_rate, i)
    print(f"视频{i}处理完毕")
