import cv2
import os
import glob

def getFiles(dir, suffix): # find the files with specific suffix
    res = []
    for root, directory, files in os.walk(dir):
        for filename in files:
            name, suf = os.path.splitext(filename)
            if suf == suffix:
                res.append(os.path.join(root, filename))
    return res


# extract frames every x frames / seconds
def extract_frames(video_path, output_path, interval, camera_i, bysec=False):
    os.makedirs(output_path, exist_ok=True)
    # open the video file
    video = cv2.VideoCapture(video_path)
 
    # get the frame rate / sec
    fps = video.get(cv2.CAP_PROP_FPS)
    # print(f"\t Frame rate is: {fps}")
 
    # calcu frame interval
    if bysec == True: 
        frame_interval = int(fps*interval)
    else:
        frame_interval = interval
    
    # count
    frame_count = 0
    num = 0
    while True:
        # get 1 frame
        ret, frame = video.read()
 
        # if no frame afterward (end of the video), break
        if not ret:
            break
 
        # get frame at the beginning of the frame_interval
        if frame_count % frame_interval == 0:
            # save path for extracted frame
            save_path = os.path.join(output_path, f"camera_{camera_i}_{num}.jpg")
            # two cameras with special placement
            if video_path == r'../data/calib/calib_0\calib_0_21334181.avi' or video_path == r'../data/calib/calib_0\calib_0_21334237.avi':
                frame_trans = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            else:
                frame_trans = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(save_path, frame_trans)
            num += 1
        # next frame
        frame_count += 1
 
    video.release()


# extract specific frames listed in the "frame_list"
def extract_spec_frames(video_path, output_path,  camera_i, frame_list=[1, 200, 250]):
    os.makedirs(output_path, exist_ok=True)
    # open the video file
    video = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        # get 1 frame
        ret, frame = video.read()

        # if no frame afterward (end of the video), break
        if not ret:
            print(f"{frame_count} frames in total.")
            break

        frame_count += 1
        # get frame at the beginning of the frame_interval
        if frame_count in frame_list:
            # save path for extracted frame
            save_path = os.path.join(output_path, f"camera_{camera_i}_{frame_count}.jpg")
            # two cameras with special placement
            if video_path == r'../data/calib/calib_0\calib_0_21334181.avi' or video_path == r'../data/calib/calib_0\calib_0_21334237.avi':
                frame_trans = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            else:
                frame_trans = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(save_path, frame_trans)

    video.release()



if __name__ == "__main__":
    # path of the videos
    directory_path = "../data/calib/calib_0"
    # video format  
    video_format = ".avi"   
    # output path    
    output_path = "../calib_frames/"


    # set the interval (frame / sec)
    # interval = 1
    # for i, file in enumerate(getFiles(directory_path, video_format)):
    #     extract_frames(file, output_path, interval, i, bysec=True)
    #     print(f"The {i}th camera finished")

    # specify the frames
    frame_list = [1, 100]
    # frame_list = [1, 50, 100, 150, 200, 250, 300]
    for i, file in enumerate(getFiles(directory_path, video_format)):
        print(file)
        extract_spec_frames(file, output_path, i, frame_list)
        print(f"The {i}th camera finished")
