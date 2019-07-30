#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import threading as thr
import numpy as np
import cv2
import math
# 引入数据库
# import human_code.Serial_Servo_Running as cmd
# import human_code.missions.commands as commands
# from human_code.missions.mission1 import through_railway

# stream = "http://127.0.0.1:8080/?action=stream?dummy=param.mjpg"
# cap = cv2.VideoCapture(stream)
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("C:\\Users\\Tonywzt\\Desktop\\赛道视频_2019_07_27\\02台阶.mp4")

ret, frame_rgb = cap.read()
frame_hsv = np.zeros((480,640,3), dtype=np.uint8)
cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV, frame_hsv)

quit_flag = False  # 用于控制子线程退出
mission_cnt = 0  # 任务计数，表示当时的任务状态


# 刷新摄像头数据
def video_cap():
    global frame_rgb, frame_hsv, ret
    global quit_flag
    print("video start")
    while cap.isOpened():
        if frame_rgb is not None and ret:
            ret, frame_rgb = cap.read()
            cv2.waitKey(30)
            if quit_flag:
                print('end of sub threat')
                break

color_ranges = {'yellow':[([16, 140, 100], [30, 230, 255])],
    'red':[([0, 50, 46],[10, 255, 255]), ([156, 50, 46], [180, 255, 255])]
}



def color_detect(frame, color_range):
    mask = np.zeros((480, 640), dtype=np.uint8)
    for (lower, upper) in color_range:
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色上限
        temp_mask = cv2.inRange(frame, lower, upper)
        # print(temp_mask.dtype)
        mask = cv2.bitwise_or(temp_mask, mask, mask=None)
    my_frame = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('color', my_frame)
    return mask

def is_through_rail():
    global frame_hsv

    mask = color_detect(frame_hsv, color_ranges['red'])
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # 找出所有轮廓
    contour_area_max = 0
    area_max_contour = None
    area = 20000
    for c in cnts:
        contour_area_temp = math.fabs(cv2.contourArea(c))
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp > area:  # 面积大于1
                area_max_contour = c
    if area_max_contour is not None:
        return True
    else :
        return False

def mission_judge():
    global mission_cnt
    if is_through_rail():
        mission_cnt = 1
    else:
        mission_cnt = 0

# 默认线程向前走
def default_thread():
    # cmd.running_action_group(0, 1)
    print("default threat")

# 过横杆线程
def through_rail_thread(the_cap):
    # through_railway(the_cap)
    print("through rail thread")

# 判断哪一个任务
def main():
    global frame_rgb, frame_hsv, ret, cap
    global quit_flag
    video_thr = thr.Thread(target=video_cap)
    video_thr.start()
    # default_thr = thr.Thread(target=default_thread)
    # through_railway_thr = thr.Thread(target=through_rail_thread, args=[cap])

    while (cap.isOpened()):
        if frame_rgb is not None and ret:
            cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV, frame_hsv)  # 更新hsv图像
            cv2.imshow('rgb', frame_rgb)  # 显示图像
            '''
            控制线程的开启，在每个线程内部都会有判断当前开启线程的状态，不成立
            则结束线程。正常情况下应阻塞线程，等待线程自行跳出再继续
            '''
            mission_judge()
            if mission_cnt == 0:
               # cmd.running_action_group(0, 1)
                print('go straight')
            elif mission_cnt==1:
                print('through rail')

            if cv2.waitKey(30) == 27:
                quit_flag = True
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
