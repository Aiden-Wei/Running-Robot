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
cap = cv2.VideoCapture("C:\\Users\\Tonywzt\\Desktop\\赛道视频_2019_07_27\\00全程视频.mp4")

ret, frame_rgb = cap.read()
frame_hsv = np.zeros((480,640,3), dtype=np.uint8)
cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV, frame_hsv)

quit_flag = False  # 用于控制子线程退出
mission_cnt = 0  # 任务计数，表示当时的任务状态
start = 1
stair = 2
#一些参数
color_ranges = {'yellow':[([16, 50, 80], [50, 255, 255])],
    'red':[([0, 50, 46],[10, 255, 255]), ([156, 50, 46], [180, 255, 255])]
}

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

# 返回特定颜色的mask, 颜色的阈值在最前面color_ranges字典中更改
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


# 判断是不是在过大门
def is_through_rail():
    global frame_hsv
    mask = color_detect(frame_hsv, color_ranges['yellow'])
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

# 判断是不是上台阶
def is_stairs():
    global frame_hsv
    mask = color_detect(frame_hsv, color_ranges['red'])
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # 找出所有轮廓
    contour_area_max = 0
    area_max_contour = None
    area = int(480*640*0.7)
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

# 判断是哪一个任务，默认为向前走
def mission_judge():
    global mission_cnt
    if is_through_rail():
        mission_cnt = start
    elif is_stairs():
        mission_cnt = stair
    else:
        mission_cnt = 0


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
            mission_judge()
            if mission_cnt == 0:
               # cmd.running_action_group(0, 1)
                print('go straight')
            elif mission_cnt==start:
                print('start')
            elif mission_cnt==stair:
                print('stair')

            if cv2.waitKey(30) == 27:
                quit_flag =     True
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
