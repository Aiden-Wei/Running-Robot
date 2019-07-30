#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import cv2
import time
import numpy as np
import math



#定义一些参数
ori_width  =  int(4*160)#原始图像640x480
ori_height =  int(3*160)
resize_width   = int(4*20)#处理图像时缩小为80x60,加快处理速度，谨慎修改！
resize_height  = int(3*20)
line_color     = (255, 0, 0)#图像显示时，画出的线框颜色
line_thickness = 2         #图像显示时，画出的线框的粗细
roi = [ # [ROI, weight]
        (0,  40,  0, 160, 0.5),
        (40, 80,  0, 160, 0.3),
        (80, 120,  0, 160, 0.2)
       ]

def edge_detection(orgimage, r_w = resize_width, r_h = resize_height, r = roi, l_c = line_color, l_t = line_thickness):
    # 缩小图像
    orgframe = cv2.resize(orgimage, (r_w, r_h), interpolation=cv2.INTER_LINEAR)
    orgframe = cv2.cvtColor(orgframe, cv2.COLOR_BGR2GRAY)  # 转化为灰度图像
    orgframe = cv2.GaussianBlur(orgframe, (3, 3), 0)  # 高斯模糊，去噪

    # 边缘检测测试
    orgframe_sobel = cv2.Sobel(orgframe, cv2.CV_16S, 0, 2)#检测水平边缘
    orgframe_sobel = cv2.convertScaleAbs(orgframe_sobel)
    cv2.imshow("orgframe_sobel", orgframe_sobel)
    # cv2.waitKey(30)


# 设置黄色的范围
color_range_yellow = [([16, 100, 100], [30, 230, 255])]

mask = None
def color_detect(frame, color_range):
    global mask
    for (lower, upper) in color_range:
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色下限

        mask = cv2.inRange(frame, lower, upper)
        frame_yellow = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("color_yellow", frame_yellow)
    return frame_yellow


def return_color(event, x, y, flags, param):
    global frame_hsv
    global cal_h, cal_s, cal_v
    if event == cv2.EVENT_LBUTTONDOWN:
        print((x,y))
        print(frame_hsv[y,x])


def getAreaMaxContour(contours, area=100):
    contour_area_max = 0
    area_max_contour = None

    for c in contours:
        contour_area_temp = math.fabs(cv2.contourArea(c))
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp > area:  # 面积大于1
                area_max_contour = c
    return area_max_contour

# cap = cv2.VideoCapture(0)
# if __name__ == "__main__":
def through_railway(cap):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frames = cap.read()
        frame_hsv = frames.copy()
        cv2.cvtColor(frames, cv2.COLOR_BGR2HSV, frame_hsv)
        if frames is not None and ret:
            # edge_detection(frames, 160, 120)
            color_detect(frame_hsv, color_range_yellow)

            cv2.setMouseCallback('frame', return_color)
        # 画出矩形轮廓
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)  # 找出所有轮廓
            cnt_large = getAreaMaxContour(cnts, area=100)  # 找到最大面积的轮廓
            if cnt_large is not None:
                rect = cv2.minAreaRect(cnt_large)  # 最小外接矩形
                box = np.int0(cv2.boxPoints(rect))  # 最小外接矩形的四个顶点
                cv2.drawContours(frames, [box], -1, (0, 0, 255, 255), 2)  # 画出四个点组成的矩形
                cv2.line(frames, (box[1, 0], box[1, 1]), (box[3, 0], box[3, 1]), line_color, line_thickness)
                cv2.line(frames, (box[0, 0], box[0, 1]), (box[2, 0], box[2, 1]), line_color, line_thickness)

            cv2.imshow("frame", frames)
            if cv2.waitKey(30)==27:
                break
            # if cnt_large is None:
            #     break
    cv2.destroyAllWindows()
