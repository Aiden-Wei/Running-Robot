#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import cv2
import time

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

cap = cv2.VideoCapture(0)

if __name__ == "__main__":
    while True:
        ret, frames = cap.read()
        if frames is not None and ret:
            edge_detection(frames, 160, 120)
            if  cv2.waitKey(1)==27:
                break
    cv2.destroyAllWindows()
