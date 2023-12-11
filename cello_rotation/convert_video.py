import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

for folder in os.listdir():
    if not os.path.isdir(folder):
        continue
    mp4=folder+ "/" + folder + ".mp4"
    print(mp4)
    output=cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*'mp4v'), 25, (640, 320))
    png_list=[]
    i=1
    while 1:
        name=folder+"/"+str(i)+".png"
        if os.path.exists(name):
            png_list.append(name)
            i+=1
        else:
            break
    for png in png_list:
        frame=cv2.imread(png)
        frame=cv2.resize(frame,(640,320))
        idx=png.replace(".png","")
        idx=idx.replace(folder+"/","")
        cv2.putText(frame,idx,(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        # cv2.imshow('frame',frame)
        # cv2.waitKey(1)
        output.write(frame)


