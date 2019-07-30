import threading as thr
import numpy as np
import cv2
import human_code.missions.commands as cmd
# 引入数据库
from human_code.missions.commands import *


# stream = "http://127.0.0.1:8080/?action=stream?dummy=param.mjpg"
# cap = cv2.VideoCapture(stream)
cap = cv2.VideoCapture(0)

ret, frame_rgb = cap.read()
frame_hsv = frame_rgb.copy()
cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV, frame_hsv)

quit_flag = False # 用于控制子线程退出
mission_cnt = 0 # 任务计数，标值当时的任务状态

def video_cap():
    global frame_rgb, frame_hsv, ret
    global quit_flag
    while True:
        if frame_rgb is not None and ret:
            ret, frame_rgb = cap.read()
            if quit_flag:
                print('end of sub threat')
                break


def main():
    global frame_rgb, frame_hsv, ret
    global quit_flag
    video_thr = thr.Thread(target=video_cap)
    video_thr.start()
    print(frame_rgb is not None)
    while True:
        if frame_rgb is not None and ret:
            cv2.imshow('rgb', frame_rgb)
            if cv2.waitKey(30) == 27:
                quit_flag = True
                break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
